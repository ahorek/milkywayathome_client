/*
 * CUDA backend for the milkyway nbody application.
 *
 * Phase 0: scaffolding for the dispatch path.
 * Phase 1: real device buffer allocation + H2D / D2H marshalling.
 * Phase 2+: kernel launchers and the per-step entry points.
 *
 * Includes are deliberately minimal: only nbody_cuda.h /
 * nbody_cuda_buffers.h (which forward-declare everything they need)
 * and the CUDA runtime header. Pulling in nbody_types.h or
 * milkyway_math.h here drags in inline asm and GCC builtins from the
 * openpa/milkyway header chain that nvcc segfaults on. Field-level
 * access to NBodyCtx/NBodyState happens in the plain-C wrapper
 * (nbody_cuda_host.c), which marshals into the POD interfaces below.
 */

#include "nbody_cuda.h"
#include "nbody_cuda_buffers.h"

#if NBODY_CUDA

#include <cuda_runtime.h>
#include <cstdlib>
#include <cstdio>
#include <climits>
#include <time.h>

/* Thrust + CUB for deterministic sorting in the Morton-code buildTree
 * path. CUB's DeviceRadixSort::SortPairs (with a Morton128 decomposer)
 * gives us radix-sort speed on a 128-bit composite key, with a
 * pre-allocated workspace so we don't pay a cudaMalloc per sort.
 *
 * milkyway_config.h (pulled in transitively via nbody_cuda.h) defines
 * `PACKED` as `__attribute__((packed))`. CUB's block_radix_rank.cuh
 * uses `PACKED` as a loop variable name, so the macro mangles its
 * declaration. We don't use PACKED in this TU; undef it before
 * including thrust/CUB. */
#ifdef PACKED
#  undef PACKED
#endif
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <cub/device/device_radix_sort.cuh>
#include <cuda/std/tuple>

/* Vendored crlibm log_rn for __device__ use. Matches CPU's mw_log()
 * bit-for-bit so per-step rounding errors don't compound chaotically.
 * Single-TU header; only included here. */
#include "nbody_cuda_crlibm.cuh"

/* nbody_types.h's NBODY_SUCCESS/NBODY_ERROR aren't visible here (we
 * can't include that header from the .cu translation unit). The values
 * are stable bit-flags in the public enum, so we mirror them locally. */
#define NBODY_CUDA_SUCCESS 0
#define NBODY_CUDA_ERROR   2     /* matches NBODY_ERROR == (1 << 1) */

/* Wrap a cudaError_t check: log the error site and return CUDA_ERROR
 * if the call failed. Used inside the alloc/upload paths so a single
 * failed CUDA call cleanly aborts the operation. */
#define CUDA_CHECK(call) do {                                         \
    cudaError_t _err = (call);                                        \
    if (_err != cudaSuccess) {                                        \
        fprintf(stderr, "[nbody_cuda] %s:%d: %s -> %s\n",             \
                __FILE__, __LINE__, #call, cudaGetErrorString(_err)); \
        return NBODY_CUDA_ERROR;                                      \
    }                                                                 \
} while (0)

/* ----- Opaque types referenced by NBodyState (nbody_types.h) ----- */

/* Phase 0 placeholders; will grow as real CUDA state is added. */
struct NBodyCUDAContext { int deviceId; };
struct NBodyCUDAKernels { int placeholder; };

/* Octree fanout — each cell has up to 8 children. Matches NSUB in
 * the OpenCL kernel header. */
#define NBODY_CUDA_NSUB 8

/* Maximum tree depth; matches MAXDEPTH default in nbody_kernels.cl. */
#ifndef NBODY_CUDA_MAXDEPTH
  #define NBODY_CUDA_MAXDEPTH 26
#endif

/* On-device tree status struct. Mirrors the OpenCL TreeStatus layout
 * (nbody_kernels.cl uses _treeStatus->{radius, bottom, blkCnt, doneCnt,
 * maxDepth, errorCode, assertionLine}). Kept POD so the host can
 * cudaMemcpy a fresh value in to reset between steps. */
struct NBodyCUDATreeStatus
{
    double radius;        /* root cell half-side */
    int    bottom;        /* lowest used cell index (grows downward in node array) */
    int    blkCnt;        /* atomic counter used by boundingBox to detect last block */
    int    doneCnt;       /* atomic counter used during summarization completion */
    int    maxDepth;      /* observed deepest tree depth */
    int    errorCode;     /* non-zero on assertion failure */
    int    assertionLine; /* source line of failing assertion */
    int    _pad;          /* keep alignment stable across host & device */
};

/* 128-bit Morton key, used by NBODY_BUILDTREE_MORTON path. Defined
 * here (rather than next to the kernels that use it) because
 * nbCUDATreeBuffersAlloc needs the full type for the CUB
 * DeviceRadixSort temp-storage query. */
struct __align__(16) Morton128
{
    unsigned long long lo;
    unsigned long long hi;
};

/* Strict less-than for Morton128 — used as the thrust::sort_by_key
 * comparator (legacy fallback path). Host+device so thrust can
 * specialise on either side. */
struct Morton128Less
{
    __host__ __device__ __forceinline__
    bool operator()(const Morton128& a, const Morton128& b) const
    {
        return (a.hi < b.hi) || (a.hi == b.hi && a.lo < b.lo);
    }
};

/* Decomposer for cub::DeviceRadixSort::SortPairs. Tuple order is
 * (hi, lo) — most-significant first. CUB's radix-sort treats the
 * tuple as a packed key with hi as MSDs, lo as LSDs. */
struct Morton128Decomposer
{
    __host__ __device__ __forceinline__
    ::cuda::std::tuple<unsigned long long&, unsigned long long&>
    operator()(Morton128& key) const
    {
        return {key.hi, key.lo};
    }
};

/* Phase 1: real device-side body storage in struct-of-arrays layout.
 * Phase 4: extended with Barnes-Hut tree storage.
 * Mirrors the OpenCL NBodyBuffers (nbody_types.h:197-248). All
 * pointers are device addresses; never dereferenced from host code.
 *
 * Body buffers (always allocated by nbCUDABuffersAlloc):
 *   pos/vel/acc/masses sized to `nbody` for exact-mode use, or to
 *   `nNode + 1` for tree-mode use (so cells reuse the same storage).
 *
 * Tree buffers (allocated separately by nbCUDATreeBuffersAlloc):
 *   tree node arrays, atomic CAS scratch, bounding-box per-SM scratch. */
struct NBodyCUDABuffers
{
    int nbody;             /* number of real bodies */
    int nNode;             /* tree-node capacity (0 if tree not allocated) */
    int numSMs;            /* device SM count, used to size BB reduction */
    int useQuad;           /* whether quadrupole buffers are present */

    /* Body / cell SoA. Sized nbody (exact) or nNode+1 (tree). */
    double* d_posX;
    double* d_posY;
    double* d_posZ;

    double* d_velX;        /* nbody only — cells have no velocity */
    double* d_velY;
    double* d_velZ;

    double* d_accX;        /* nbody only — cells have no acceleration */
    double* d_accY;
    double* d_accZ;

    double* d_masses;      /* sized nbody (exact) or nNode+1 (tree) */

    /* Tree-side buffers. NULL when not allocated. */
    double* d_minX;        /* per-SM bounding-box reduction scratch */
    double* d_minY;
    double* d_minZ;
    double* d_maxX;
    double* d_maxY;
    double* d_maxZ;

    int*    d_start;       /* per-cell start index (size nNode+1) */
    int*    d_count;       /* per-cell body count (size nNode+1) */
    int*    d_sort;        /* sort permutation (size nbody) */
    int*    d_child;       /* NSUB children per node (size NSUB*(nNode+1)) */
    double* d_critRadii;   /* per-cell opening criterion radius (size nNode+1) */

    double* d_quadXX;      /* quadrupole moments (each size nNode+1) */
    double* d_quadXY;
    double* d_quadXZ;
    double* d_quadYY;
    double* d_quadYZ;
    double* d_quadZZ;

    /* Packed quadrupole moments: 8 doubles per cell (6 quad components
     * + 2 pad) so cell N's full quad fits in a single 64-byte cache
     * line at offset N*8. Populated by nbCUDAQuadPackKernel after
     * QuadMoments completes; read by ForceTree instead of the 6
     * separate d_quad** arrays — coalesces 6 cache-line loads per
     * accepted-cell visit into 1. Sized (nNode+1) * 8 doubles. */
    double* d_quadPacked;

    struct NBodyCUDATreeStatus* d_treeStatus;  /* size 1 */

    /* ----- Morton-based deterministic buildTree scratch -----
     * Allocated alongside other tree buffers in nbCUDATreeBuffersAlloc
     * when NBODY_BUILDTREE_MORTON is enabled at compile time, or always
     * (small footprint) so a runtime flag can choose at launch time. */
    Morton128* d_morton;              /* per-body 128-bit Morton key, size nbody */
    int*    d_sortedIdx;              /* sort permutation, size nbody */

    /* Per-level work queues, double-buffered. Max cells per level is
     * bounded by nbody (a level can't have more cells than bodies) and
     * by nNode - nbody (total cell budget). Size at nbody for safety. */
    int*    d_lvlInCells;
    int*    d_lvlInStart;
    int*    d_lvlInEnd;
    int*    d_lvlInDepth;
    int*    d_lvlOutCells;
    int*    d_lvlOutStart;
    int*    d_lvlOutEnd;
    int*    d_lvlOutDepth;

    /* Fused-Morton path: device-side level counters (in_count, out_count).
     * No host sync needed between levels — the swap kernel handles it. */
    int*    d_lvlInCount;             /* 1 int */
    int*    d_lvlOutCount;            /* 1 int */

    /* CUB DeviceRadixSort scratch (opt #23). ComputeMorton writes to
     * the *Scratch arrays; CUB sorts those into d_morton/d_sortedIdx
     * with d_sortTempStorage as workspace. Pre-allocated (sized at
     * tree-buffers init time), so no per-step cudaMalloc. */
    Morton128* d_mortonScratch;       /* sized as d_morton */
    int*       d_sortedIdxScratch;    /* sized as d_sortedIdx */
    void*      d_sortTempStorage;     /* size queried via CUB */
    size_t     sortTempBytes;

    /* CUDA-graph capture for the Morton tree-build pipeline (opt #24).
     * Captured ONCE on the first nbCUDABuildTreeMorton call and
     * replayed every subsequent step, eliminating the ~50 launches'
     * worth of host-side launch overhead per build. mortonStream is
     * dedicated to the captured ops; we cudaStreamSynchronize after
     * each cudaGraphLaunch so the rest of the per-step pipeline
     * (still on the default stream) sees the new tree. */
    cudaStream_t   mortonStream;
    cudaGraph_t    mortonGraph;
    cudaGraphExec_t mortonGraphExec;
    int            mortonGraphCaptured; /* 0 until first capture done */

    /* opt #8 elaborate — pinned host buffer + async D2H stream for
     * bestLikelihood eval. The bestLike window evaluates the per-step
     * likelihood, which requires the CPU to see the latest body
     * positions/velocities. Sync nbCUDAMarshalBodiesFromDevice stalls
     * the GPU pipeline; async D2H lets the next step's GPU kernels
     * overlap the previous step's D2H + EMD compute on the host.
     *
     * Layout of h_pinnedBodies (6*nbody doubles, pinned, contiguous):
     *   posX[0..nbody)  posY[0..nbody)  posZ[0..nbody)
     *   velX[0..nbody)  velY[0..nbody)  velZ[0..nbody)
     *
     * The previous-step's likelihood eval reads from this buffer;
     * the current-step's async D2H writes into it (no overlap of
     * the two — bestLikePending tracks ownership). */
    double*        h_pinnedBodies;     /* 6 * nbody doubles, pinned */
    cudaStream_t   bestLikeStream;
    cudaEvent_t    bestLikeEvent;
    int            bestLikePending;    /* 1 if an async D2H is in flight */
};

/* ----- Phase 5c: device introspection helper for the host wrapper ----- */

extern "C" int nbCUDABuffersGetNbody(const struct NBodyCUDABuffers* buffers)
{
    return buffers ? buffers->nbody : 0;
}

/* Read d_treeStatus->maxDepth back to host. Used for diagnostics
 * (e.g. to check whether the Morton path's rank-partition fallback
 * at depth >20 ever fires for a given WU). Returns -1 on error. */
extern "C" int nbCUDABuffersGetMaxDepth(const struct NBodyCUDABuffers* buffers)
{
    if (!buffers || !buffers->d_treeStatus) return -1;
    int maxDepth = -1;
    cudaError_t e = cudaMemcpy(&maxDepth, &buffers->d_treeStatus->maxDepth,
                               sizeof(int), cudaMemcpyDeviceToHost);
    if (e != cudaSuccess) return -1;
    return maxDepth;
}

extern "C" int nbCUDABuffersGetNNode(const struct NBodyCUDABuffers* buffers)
{
    return buffers ? buffers->nNode : 0;
}

extern "C" int nbCUDAGetDeviceSMCount(int* outSMs)
{
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "[nbody_cuda] cudaGetDeviceCount: %s (%d)\n",
                cudaGetErrorString(err), (int) err);
        return -1;
    }
    if (count <= 0)
    {
        fprintf(stderr, "[nbody_cuda] no CUDA devices reported (count=%d)\n", count);
        return -1;
    }
    int dev = 0;
    err = cudaGetDevice(&dev);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "[nbody_cuda] cudaGetDevice: %s\n", cudaGetErrorString(err));
        return -1;
    }
    cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop, dev);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "[nbody_cuda] cudaGetDeviceProperties: %s\n", cudaGetErrorString(err));
        return -1;
    }
    fprintf(stderr, "[nbody_cuda] device %d: %s, SMs=%d, compute=%d.%d\n",
            dev, prop.name, prop.multiProcessorCount, prop.major, prop.minor);
    if (outSMs) *outSMs = prop.multiProcessorCount;
    return 0;
}

/* DEBUG dump: walk the CUDA tree DFS-style and write per-cell
 * (mass, cx, cy, cz, rcrit2) records to a binary file. Used by the
 * NBODY_DUMP_TREE diagnostic to compare CoMs/Rcrit2 vs CPU's tree. */
extern "C" int nbCUDABuffersDumpTree(const struct NBodyCUDABuffers* buffers,
                                     int nbody, int nNode, const char* path)
{
    if (!buffers || !path) return -1;
    int totalNodes = nNode + 1;
    int childArrSize = NBODY_CUDA_NSUB * totalNodes;
    double* h_posX = (double*) malloc(sizeof(double) * (size_t) totalNodes);
    double* h_posY = (double*) malloc(sizeof(double) * (size_t) totalNodes);
    double* h_posZ = (double*) malloc(sizeof(double) * (size_t) totalNodes);
    double* h_mass = (double*) malloc(sizeof(double) * (size_t) totalNodes);
    double* h_rc2  = (double*) malloc(sizeof(double) * (size_t) totalNodes);
    int*    h_chld = (int*)    malloc(sizeof(int)    * (size_t) childArrSize);
    int rc = -1;
    if (h_posX && h_posY && h_posZ && h_mass && h_rc2 && h_chld)
    {
        cudaMemcpy(h_posX, buffers->d_posX, sizeof(double)*totalNodes, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_posY, buffers->d_posY, sizeof(double)*totalNodes, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_posZ, buffers->d_posZ, sizeof(double)*totalNodes, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_mass, buffers->d_masses, sizeof(double)*totalNodes, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_rc2,  buffers->d_critRadii, sizeof(double)*totalNodes, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_chld, buffers->d_child, sizeof(int)*childArrSize, cudaMemcpyDeviceToHost);

        FILE* f = fopen(path, "wb");
        if (f)
        {
            int* stkCell = (int*) malloc(sizeof(int) * 64);
            int* stkPos  = (int*) malloc(sizeof(int) * 64);
            int top = 0;
            int nrecs = 0;
            stkCell[0] = nNode;
            stkPos[0]  = 0;
            int dupCount = 0;
            int parentOf[200000]; /* tracks parent of each visited cell */
            for (int z = 0; z < 200000; ++z) parentOf[z] = -1;
            {
                int idx = nNode;
                double m = h_mass[idx], cx=h_posX[idx], cy=h_posY[idx], cz=h_posZ[idx], r=h_rc2[idx];
                fwrite(&m, sizeof(double),1,f); fwrite(&cx,sizeof(double),1,f);
                fwrite(&cy,sizeof(double),1,f); fwrite(&cz,sizeof(double),1,f);
                fwrite(&r, sizeof(double),1,f);
                ++nrecs;
                parentOf[idx] = idx;
            }
            while (top >= 0)
            {
                int cell = stkCell[top];
                int p    = stkPos[top];
                if (p >= NBODY_CUDA_NSUB) { --top; continue; }
                int ch = h_chld[NBODY_CUDA_NSUB * cell + p];
                stkPos[top] = p + 1;
                if (ch < 0) { --top; continue; }
                if (ch >= nbody)
                {
                    if (ch < 200000 && parentOf[ch] != -1) {
                        ++dupCount;
                        if (dupCount <= 5) {
                            fprintf(stderr, "[DUP] cell %d already visited via parent %d, now via parent %d\n",
                                    ch, parentOf[ch], cell);
                        }
                    } else if (ch < 200000) {
                        parentOf[ch] = cell;
                    }
                    double m=h_mass[ch], cx=h_posX[ch], cy=h_posY[ch], cz=h_posZ[ch], r=h_rc2[ch];
                    fwrite(&m,sizeof(double),1,f); fwrite(&cx,sizeof(double),1,f);
                    fwrite(&cy,sizeof(double),1,f); fwrite(&cz,sizeof(double),1,f);
                    fwrite(&r, sizeof(double),1,f);
                    ++nrecs;
                    ++top;
                    stkCell[top] = ch;
                    stkPos[top]  = 0;
                }
            }
            fprintf(stderr, "[DEBUG] tree dup count: %d\n", dupCount);
            fclose(f);
            fprintf(stderr, "[DEBUG] dumped %d CUDA tree-cell records to %s\n", nrecs, path);
            free(stkCell); free(stkPos);
            rc = nrecs;
        }
    }
    free(h_posX); free(h_posY); free(h_posZ); free(h_mass); free(h_rc2); free(h_chld);
    return rc;
}

/* DEBUG: same as DumpTree but also includes quadXX..ZZ per cell. */
extern "C" int nbCUDABuffersDumpTreeQuad(const struct NBodyCUDABuffers* buffers,
                                         int nbody, int nNode, const char* path)
{
    if (!buffers || !path) return -1;
    int totalNodes = nNode + 1;
    int childArrSize = NBODY_CUDA_NSUB * totalNodes;
    double* h_posX = (double*) malloc(sizeof(double) * (size_t) totalNodes);
    double* h_posY = (double*) malloc(sizeof(double) * (size_t) totalNodes);
    double* h_posZ = (double*) malloc(sizeof(double) * (size_t) totalNodes);
    double* h_mass = (double*) malloc(sizeof(double) * (size_t) totalNodes);
    double* h_rc2  = (double*) malloc(sizeof(double) * (size_t) totalNodes);
    double* h_qxx  = (double*) malloc(sizeof(double) * (size_t) totalNodes);
    double* h_qxy  = (double*) malloc(sizeof(double) * (size_t) totalNodes);
    double* h_qxz  = (double*) malloc(sizeof(double) * (size_t) totalNodes);
    double* h_qyy  = (double*) malloc(sizeof(double) * (size_t) totalNodes);
    double* h_qyz  = (double*) malloc(sizeof(double) * (size_t) totalNodes);
    double* h_qzz  = (double*) malloc(sizeof(double) * (size_t) totalNodes);
    int*    h_chld = (int*)    malloc(sizeof(int)    * (size_t) childArrSize);
    int rc = -1;
    if (h_posX && h_posY && h_posZ && h_mass && h_rc2 && h_chld &&
        h_qxx && h_qxy && h_qxz && h_qyy && h_qyz && h_qzz)
    {
        cudaMemcpy(h_posX, buffers->d_posX, sizeof(double)*totalNodes, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_posY, buffers->d_posY, sizeof(double)*totalNodes, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_posZ, buffers->d_posZ, sizeof(double)*totalNodes, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_mass, buffers->d_masses, sizeof(double)*totalNodes, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_rc2,  buffers->d_critRadii, sizeof(double)*totalNodes, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_chld, buffers->d_child, sizeof(int)*childArrSize, cudaMemcpyDeviceToHost);
        if (buffers->d_quadXX) {
            cudaMemcpy(h_qxx, buffers->d_quadXX, sizeof(double)*totalNodes, cudaMemcpyDeviceToHost);
            cudaMemcpy(h_qxy, buffers->d_quadXY, sizeof(double)*totalNodes, cudaMemcpyDeviceToHost);
            cudaMemcpy(h_qxz, buffers->d_quadXZ, sizeof(double)*totalNodes, cudaMemcpyDeviceToHost);
            cudaMemcpy(h_qyy, buffers->d_quadYY, sizeof(double)*totalNodes, cudaMemcpyDeviceToHost);
            cudaMemcpy(h_qyz, buffers->d_quadYZ, sizeof(double)*totalNodes, cudaMemcpyDeviceToHost);
            cudaMemcpy(h_qzz, buffers->d_quadZZ, sizeof(double)*totalNodes, cudaMemcpyDeviceToHost);
        }
        FILE* f = fopen(path, "wb");
        if (f) {
            int* stkCell = (int*) malloc(sizeof(int) * 64);
            int* stkPos  = (int*) malloc(sizeof(int) * 64);
            int top = 0; int nrecs = 0;
            stkCell[0] = nNode; stkPos[0] = 0;
            #define EMIT(idx) do { \
                double m=h_mass[idx], cx=h_posX[idx], cy=h_posY[idx], cz=h_posZ[idx], r=h_rc2[idx]; \
                double qxx=h_qxx[idx],qxy=h_qxy[idx],qxz=h_qxz[idx],qyy=h_qyy[idx],qyz=h_qyz[idx],qzz=h_qzz[idx]; \
                fwrite(&m,sizeof(double),1,f); fwrite(&cx,sizeof(double),1,f); \
                fwrite(&cy,sizeof(double),1,f); fwrite(&cz,sizeof(double),1,f); \
                fwrite(&r,sizeof(double),1,f); \
                fwrite(&qxx,sizeof(double),1,f); fwrite(&qxy,sizeof(double),1,f); \
                fwrite(&qxz,sizeof(double),1,f); fwrite(&qyy,sizeof(double),1,f); \
                fwrite(&qyz,sizeof(double),1,f); fwrite(&qzz,sizeof(double),1,f); \
                ++nrecs; \
            } while(0)
            EMIT(nNode);
            while (top >= 0) {
                int cell = stkCell[top]; int p = stkPos[top];
                if (p >= NBODY_CUDA_NSUB) { --top; continue; }
                int ch = h_chld[NBODY_CUDA_NSUB * cell + p];
                stkPos[top] = p + 1;
                if (ch < 0) { --top; continue; }
                if (ch >= nbody) {
                    EMIT(ch);
                    ++top; stkCell[top] = ch; stkPos[top] = 0;
                }
            }
            #undef EMIT
            fclose(f);
            fprintf(stderr, "[DEBUG] dumped %d CUDA tree+quad records to %s\n", nrecs, path);
            free(stkCell); free(stkPos);
            rc = nrecs;
        }
    }
    free(h_posX); free(h_posY); free(h_posZ); free(h_mass); free(h_rc2); free(h_chld);
    free(h_qxx); free(h_qxy); free(h_qxz); free(h_qyy); free(h_qyz); free(h_qzz);
    return rc;
}

/* The lifecycle entry points (nbInitCUDA / nbReleaseCUDA /
 * nbStepSystemCUDA / nbRunSystemCUDA) live in nbody_cuda_host.c.
 * They need access to NBodyState/NBodyCtx fields, which the .cu
 * translation unit can't safely include. The host wrapper uses the
 * POD-only buffer + launcher API exported here. */

/* ----- Phase 1: device buffer alloc / free / upload / download ----- */

extern "C" int nbCUDABuffersComputeNNode(int nbody, int numSMs)
{
    /* Mirrors nbFindNNode in nbody_cl.c:2056. Round up to a multiple
     * of warpSize (32) so per-warp tree-walk loops don't underrun. */
    int nNode = 2 * nbody;
    if (nNode < 1024 * numSMs) nNode = 1024 * numSMs;
    nNode = (nNode + 31) & ~31;
    return nNode;
}

extern "C" NBodyStatus_int nbCUDABuffersAlloc(struct NBodyCUDABuffers** outBuffers,
                                              int nbody,
                                              int nNode)
{
    if (!outBuffers || nbody <= 0 || nNode < 0)
    {
        return NBODY_CUDA_ERROR;
    }
    *outBuffers = NULL;

    NBodyCUDABuffers* b = (NBodyCUDABuffers*) calloc(1, sizeof(NBodyCUDABuffers));
    if (!b)
    {
        return NBODY_CUDA_ERROR;
    }
    b->nbody  = nbody;
    b->nNode  = 0;        /* set non-zero only by nbCUDATreeBuffersAlloc */
    b->numSMs = 0;
    b->useQuad = 0;

    /* pos/mass need to hold both bodies AND tree cells when nNode>0.
     * vel/acc are body-only — tree cells have no kinematic state. */
    const size_t bodyBytes = (size_t) nbody * sizeof(double);
    const size_t posBytes  = (nNode > 0)
                           ? ((size_t) (nNode + 1) * sizeof(double))
                           : bodyBytes;

    /* Allocate the 10 body buffers. On any failure, roll back
     * everything allocated so far so the caller never sees a
     * half-initialized buffer pack. */
    cudaError_t err = cudaSuccess;
    struct AllocSpec { double** ptr; size_t bytes; } specs[10] = {
        { &b->d_posX,   posBytes  }, { &b->d_posY,   posBytes  }, { &b->d_posZ,   posBytes  },
        { &b->d_velX,   bodyBytes }, { &b->d_velY,   bodyBytes }, { &b->d_velZ,   bodyBytes },
        { &b->d_accX,   bodyBytes }, { &b->d_accY,   bodyBytes }, { &b->d_accZ,   bodyBytes },
        { &b->d_masses, posBytes  }
    };
    int allocated = 0;
    for (; allocated < 10; ++allocated)
    {
        err = cudaMalloc((void**) specs[allocated].ptr, specs[allocated].bytes);
        if (err != cudaSuccess) break;
    }
    if (err != cudaSuccess)
    {
        fprintf(stderr, "[nbody_cuda] cudaMalloc failed at buffer %d/%d: %s\n",
                allocated, 10, cudaGetErrorString(err));
        for (int i = 0; i < allocated; ++i)
        {
            cudaFree(*specs[i].ptr);
            *specs[i].ptr = NULL;
        }
        free(b);
        return NBODY_CUDA_ERROR;
    }

    *outBuffers = b;
    return NBODY_CUDA_SUCCESS;
}

/* Helper: cudaMalloc + register for rollback. Returns 0 on success. */
static int nbCUDAMallocRecorded(void** ptr, size_t bytes,
                                void** allocList, int* nAlloc, int maxAlloc)
{
    cudaError_t err = cudaMalloc(ptr, bytes);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "[nbody_cuda] cudaMalloc(%zu) failed: %s\n",
                bytes, cudaGetErrorString(err));
        return -1;
    }
    if (*nAlloc >= maxAlloc)
    {
        fprintf(stderr, "[nbody_cuda] alloc-tracking overflow\n");
        cudaFree(*ptr);
        *ptr = NULL;
        return -1;
    }
    allocList[*nAlloc] = *ptr;
    (*nAlloc)++;
    return 0;
}

extern "C" NBodyStatus_int nbCUDATreeBuffersAlloc(struct NBodyCUDABuffers* buffers,
                                                  int nNode,
                                                  int numSMs,
                                                  int useQuad)
{
    if (!buffers || nNode <= 0 || numSMs <= 0) return NBODY_CUDA_ERROR;
    if (buffers->d_treeStatus) return NBODY_CUDA_ERROR;  /* already allocated */

    buffers->nNode   = nNode;
    buffers->numSMs  = numSMs;
    buffers->useQuad = useQuad;

    const size_t nodeR  = (size_t) (nNode + 1) * sizeof(double);
    const size_t nodeI  = (size_t) (nNode + 1) * sizeof(int);
    const size_t childR = (size_t) NBODY_CUDA_NSUB * (size_t) (nNode + 1) * sizeof(int);
    const size_t bbR    = (size_t) numSMs * sizeof(double);
    const size_t bodyI  = (size_t) buffers->nbody * sizeof(int);

    /* Track allocations for rollback on failure. Max we'll allocate
     * is 6 BB + 4 tree-int + 1 critRadii + 6 quad + 1 status + 14
     * Morton-path = 32. Use 48 for headroom. */
    void* allocList[48];
    int   nAlloc = 0;

    /* Bounding-box per-SM scratch. */
    if (nbCUDAMallocRecorded((void**) &buffers->d_minX, bbR, allocList, &nAlloc, 48)) goto fail;
    if (nbCUDAMallocRecorded((void**) &buffers->d_minY, bbR, allocList, &nAlloc, 48)) goto fail;
    if (nbCUDAMallocRecorded((void**) &buffers->d_minZ, bbR, allocList, &nAlloc, 48)) goto fail;
    if (nbCUDAMallocRecorded((void**) &buffers->d_maxX, bbR, allocList, &nAlloc, 48)) goto fail;
    if (nbCUDAMallocRecorded((void**) &buffers->d_maxY, bbR, allocList, &nAlloc, 48)) goto fail;
    if (nbCUDAMallocRecorded((void**) &buffers->d_maxZ, bbR, allocList, &nAlloc, 48)) goto fail;

    /* Per-cell ints. */
    if (nbCUDAMallocRecorded((void**) &buffers->d_start, nodeI,  allocList, &nAlloc, 48)) goto fail;
    if (nbCUDAMallocRecorded((void**) &buffers->d_count, nodeI,  allocList, &nAlloc, 48)) goto fail;
    if (nbCUDAMallocRecorded((void**) &buffers->d_sort,  bodyI,  allocList, &nAlloc, 48)) goto fail;
    if (nbCUDAMallocRecorded((void**) &buffers->d_child, childR, allocList, &nAlloc, 48)) goto fail;

    /* Per-cell opening criterion radius. */
    if (nbCUDAMallocRecorded((void**) &buffers->d_critRadii, nodeR, allocList, &nAlloc, 48)) goto fail;

    /* Quadrupole moments (optional). */
    if (useQuad)
    {
        if (nbCUDAMallocRecorded((void**) &buffers->d_quadXX, nodeR, allocList, &nAlloc, 48)) goto fail;
        if (nbCUDAMallocRecorded((void**) &buffers->d_quadXY, nodeR, allocList, &nAlloc, 48)) goto fail;
        if (nbCUDAMallocRecorded((void**) &buffers->d_quadXZ, nodeR, allocList, &nAlloc, 48)) goto fail;
        if (nbCUDAMallocRecorded((void**) &buffers->d_quadYY, nodeR, allocList, &nAlloc, 48)) goto fail;
        if (nbCUDAMallocRecorded((void**) &buffers->d_quadYZ, nodeR, allocList, &nAlloc, 48)) goto fail;
        if (nbCUDAMallocRecorded((void**) &buffers->d_quadZZ, nodeR, allocList, &nAlloc, 48)) goto fail;
        /* Packed quad: (nNode+1) * 8 doubles. ~5 MB for 80K cells. */
        const size_t packBytes = (size_t) (nNode + 1) * 8 * sizeof(double);
        if (nbCUDAMallocRecorded((void**) &buffers->d_quadPacked, packBytes, allocList, &nAlloc, 48)) goto fail;
    }

    /* Tree status struct. */
    if (nbCUDAMallocRecorded((void**) &buffers->d_treeStatus,
                             sizeof(struct NBodyCUDATreeStatus),
                             allocList, &nAlloc, 48)) goto fail;

    /* Zero-init the status struct. */
    if (cudaMemset(buffers->d_treeStatus, 0,
                   sizeof(struct NBodyCUDATreeStatus)) != cudaSuccess)
    {
        goto fail;
    }

    /* ----- Morton-path scratch ----- */
    {
        /* d_morton holds Morton128 (16 bytes each); other queues are
         * int (4 bytes). The buffer struct's d_morton field is typed
         * as `Morton128*` but Morton128's definition is later in this
         * TU — cudaMalloc just needs the byte count and a void**. */
        const size_t nbodyU128 = (size_t) buffers->nbody * 16;
        const size_t lvlI     = (size_t) buffers->nbody * sizeof(int);
        if (nbCUDAMallocRecorded((void**) &buffers->d_morton,           nbodyU128, allocList, &nAlloc, 48)) goto fail;
        if (nbCUDAMallocRecorded((void**) &buffers->d_sortedIdx,        lvlI,     allocList, &nAlloc, 48)) goto fail;
        if (nbCUDAMallocRecorded((void**) &buffers->d_mortonScratch,    nbodyU128, allocList, &nAlloc, 48)) goto fail;
        if (nbCUDAMallocRecorded((void**) &buffers->d_sortedIdxScratch, lvlI,     allocList, &nAlloc, 48)) goto fail;
        if (nbCUDAMallocRecorded((void**) &buffers->d_lvlInCells,       lvlI,     allocList, &nAlloc, 48)) goto fail;
        if (nbCUDAMallocRecorded((void**) &buffers->d_lvlInStart,       lvlI,     allocList, &nAlloc, 48)) goto fail;
        if (nbCUDAMallocRecorded((void**) &buffers->d_lvlInEnd,         lvlI,     allocList, &nAlloc, 48)) goto fail;
        if (nbCUDAMallocRecorded((void**) &buffers->d_lvlInDepth,       lvlI,     allocList, &nAlloc, 48)) goto fail;
        if (nbCUDAMallocRecorded((void**) &buffers->d_lvlOutCells,      lvlI,     allocList, &nAlloc, 48)) goto fail;
        if (nbCUDAMallocRecorded((void**) &buffers->d_lvlOutStart,      lvlI,     allocList, &nAlloc, 48)) goto fail;
        if (nbCUDAMallocRecorded((void**) &buffers->d_lvlOutEnd,        lvlI,     allocList, &nAlloc, 48)) goto fail;
        if (nbCUDAMallocRecorded((void**) &buffers->d_lvlOutDepth,      lvlI,     allocList, &nAlloc, 48)) goto fail;
        if (nbCUDAMallocRecorded((void**) &buffers->d_lvlInCount,       sizeof(int), allocList, &nAlloc, 48)) goto fail;
        if (nbCUDAMallocRecorded((void**) &buffers->d_lvlOutCount,      sizeof(int), allocList, &nAlloc, 48)) goto fail;

        /* CUB DeviceRadixSort temp-storage size depends on N and the
         * key type. Query once (passing nullptr for d_temp_storage)
         * and pre-allocate. */
        buffers->sortTempBytes = 0;
        cudaError_t cubQueryErr = cub::DeviceRadixSort::SortPairs(
            (void*) nullptr, buffers->sortTempBytes,
            (const Morton128*) buffers->d_mortonScratch,
            (Morton128*)       buffers->d_morton,
            (const int*)       buffers->d_sortedIdxScratch,
            (int*)             buffers->d_sortedIdx,
            buffers->nbody,
            Morton128Decomposer{},
            0, 128);
        if (cubQueryErr != cudaSuccess || buffers->sortTempBytes == 0)
        {
            fprintf(stderr, "[nbody_cuda] cub::DeviceRadixSort::SortPairs query failed (%s, %zu bytes)\n",
                    cudaGetErrorString(cubQueryErr), buffers->sortTempBytes);
            goto fail;
        }
        if (nbCUDAMallocRecorded((void**) &buffers->d_sortTempStorage,
                                 buffers->sortTempBytes,
                                 allocList, &nAlloc, 48)) goto fail;

        /* opt #24: dedicated stream for CUDA-graph capture of the
         * Morton tree-build pipeline. Created here, captured on the
         * first nbCUDABuildTreeMorton call, replayed each step. */
        if (cudaStreamCreate(&buffers->mortonStream) != cudaSuccess)
        {
            buffers->mortonStream = NULL;
            goto fail;
        }

        /* opt #8 elaborate: pinned host buffer (sized 6*nbody doubles)
         * + dedicated stream + completion event for async body D2H.
         * Failure here is fatal but cleanup is shared with the
         * outer fail label. */
        const size_t pinnedBytes = (size_t) 6 * buffers->nbody * sizeof(double);
        if (cudaMallocHost((void**) &buffers->h_pinnedBodies, pinnedBytes) != cudaSuccess)
        {
            buffers->h_pinnedBodies = NULL;
            goto fail;
        }
        if (cudaStreamCreate(&buffers->bestLikeStream) != cudaSuccess)
        {
            buffers->bestLikeStream = NULL;
            goto fail;
        }
        if (cudaEventCreateWithFlags(&buffers->bestLikeEvent,
                                     cudaEventDisableTiming) != cudaSuccess)
        {
            buffers->bestLikeEvent = NULL;
            goto fail;
        }
    }

    return NBODY_CUDA_SUCCESS;

fail:
    for (int i = 0; i < nAlloc; ++i) cudaFree(allocList[i]);
    /* Null out everything we touched so a later free is safe. */
    buffers->d_minX = buffers->d_minY = buffers->d_minZ = NULL;
    buffers->d_maxX = buffers->d_maxY = buffers->d_maxZ = NULL;
    buffers->d_start = buffers->d_count = buffers->d_sort = buffers->d_child = NULL;
    buffers->d_critRadii = NULL;
    buffers->d_quadXX = buffers->d_quadXY = buffers->d_quadXZ = NULL;
    buffers->d_quadYY = buffers->d_quadYZ = buffers->d_quadZZ = NULL;
    buffers->d_quadPacked = NULL;
    buffers->d_treeStatus = NULL;
    buffers->d_morton = NULL;
    buffers->d_sortedIdx = NULL;
    buffers->d_lvlInCells = buffers->d_lvlInStart = buffers->d_lvlInEnd = buffers->d_lvlInDepth = NULL;
    buffers->d_lvlOutCells = buffers->d_lvlOutStart = buffers->d_lvlOutEnd = buffers->d_lvlOutDepth = NULL;
    buffers->d_lvlInCount = buffers->d_lvlOutCount = NULL;
    buffers->d_mortonScratch = NULL;
    buffers->d_sortedIdxScratch = NULL;
    buffers->d_sortTempStorage = NULL;
    buffers->sortTempBytes = 0;
    buffers->mortonStream = NULL;
    buffers->mortonGraph = NULL;
    buffers->mortonGraphExec = NULL;
    buffers->mortonGraphCaptured = 0;
    buffers->h_pinnedBodies = NULL;
    buffers->bestLikeStream = NULL;
    buffers->bestLikeEvent = NULL;
    buffers->bestLikePending = 0;
    buffers->nNode = 0;
    return NBODY_CUDA_ERROR;
}

extern "C" void nbCUDABuffersFree(struct NBodyCUDABuffers* buffers)
{
    if (!buffers) return;
    /* Body buffers. */
    if (buffers->d_posX)   cudaFree(buffers->d_posX);
    if (buffers->d_posY)   cudaFree(buffers->d_posY);
    if (buffers->d_posZ)   cudaFree(buffers->d_posZ);
    if (buffers->d_velX)   cudaFree(buffers->d_velX);
    if (buffers->d_velY)   cudaFree(buffers->d_velY);
    if (buffers->d_velZ)   cudaFree(buffers->d_velZ);
    if (buffers->d_accX)   cudaFree(buffers->d_accX);
    if (buffers->d_accY)   cudaFree(buffers->d_accY);
    if (buffers->d_accZ)   cudaFree(buffers->d_accZ);
    if (buffers->d_masses) cudaFree(buffers->d_masses);
    /* Tree buffers (NULL-safe). */
    if (buffers->d_minX) cudaFree(buffers->d_minX);
    if (buffers->d_minY) cudaFree(buffers->d_minY);
    if (buffers->d_minZ) cudaFree(buffers->d_minZ);
    if (buffers->d_maxX) cudaFree(buffers->d_maxX);
    if (buffers->d_maxY) cudaFree(buffers->d_maxY);
    if (buffers->d_maxZ) cudaFree(buffers->d_maxZ);
    if (buffers->d_start) cudaFree(buffers->d_start);
    if (buffers->d_count) cudaFree(buffers->d_count);
    if (buffers->d_sort)  cudaFree(buffers->d_sort);
    if (buffers->d_child) cudaFree(buffers->d_child);
    if (buffers->d_critRadii) cudaFree(buffers->d_critRadii);
    if (buffers->d_quadXX) cudaFree(buffers->d_quadXX);
    if (buffers->d_quadXY) cudaFree(buffers->d_quadXY);
    if (buffers->d_quadXZ) cudaFree(buffers->d_quadXZ);
    if (buffers->d_quadYY) cudaFree(buffers->d_quadYY);
    if (buffers->d_quadYZ) cudaFree(buffers->d_quadYZ);
    if (buffers->d_quadZZ) cudaFree(buffers->d_quadZZ);
    if (buffers->d_quadPacked) cudaFree(buffers->d_quadPacked);
    if (buffers->d_treeStatus) cudaFree(buffers->d_treeStatus);
    /* Morton-path scratch (NULL-safe). */
    if (buffers->d_morton)             cudaFree(buffers->d_morton);
    if (buffers->d_sortedIdx)          cudaFree(buffers->d_sortedIdx);
    if (buffers->d_lvlInCells)         cudaFree(buffers->d_lvlInCells);
    if (buffers->d_lvlInStart)         cudaFree(buffers->d_lvlInStart);
    if (buffers->d_lvlInEnd)           cudaFree(buffers->d_lvlInEnd);
    if (buffers->d_lvlInDepth)         cudaFree(buffers->d_lvlInDepth);
    if (buffers->d_lvlOutCells)        cudaFree(buffers->d_lvlOutCells);
    if (buffers->d_lvlOutStart)        cudaFree(buffers->d_lvlOutStart);
    if (buffers->d_lvlOutEnd)          cudaFree(buffers->d_lvlOutEnd);
    if (buffers->d_lvlOutDepth)        cudaFree(buffers->d_lvlOutDepth);
    if (buffers->d_lvlInCount)         cudaFree(buffers->d_lvlInCount);
    if (buffers->d_lvlOutCount)        cudaFree(buffers->d_lvlOutCount);
    if (buffers->d_mortonScratch)      cudaFree(buffers->d_mortonScratch);
    if (buffers->d_sortedIdxScratch)   cudaFree(buffers->d_sortedIdxScratch);
    if (buffers->d_sortTempStorage)    cudaFree(buffers->d_sortTempStorage);
    if (buffers->mortonGraphExec)      cudaGraphExecDestroy(buffers->mortonGraphExec);
    if (buffers->mortonGraph)          cudaGraphDestroy(buffers->mortonGraph);
    if (buffers->mortonStream)         cudaStreamDestroy(buffers->mortonStream);
    if (buffers->h_pinnedBodies)       cudaFreeHost(buffers->h_pinnedBodies);
    if (buffers->bestLikeStream)       cudaStreamDestroy(buffers->bestLikeStream);
    if (buffers->bestLikeEvent)        cudaEventDestroy(buffers->bestLikeEvent);
    free(buffers);
}

extern "C" NBodyStatus_int nbCUDABuffersUploadBodies(struct NBodyCUDABuffers* buffers,
                                                     const double* hPosX,
                                                     const double* hPosY,
                                                     const double* hPosZ,
                                                     const double* hVelX,
                                                     const double* hVelY,
                                                     const double* hVelZ,
                                                     const double* hMasses,
                                                     int nbody)
{
    if (!buffers || nbody != buffers->nbody) return NBODY_CUDA_ERROR;
    const size_t bytes = (size_t) nbody * sizeof(double);

    CUDA_CHECK(cudaMemcpy(buffers->d_posX, hPosX, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(buffers->d_posY, hPosY, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(buffers->d_posZ, hPosZ, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(buffers->d_velX, hVelX, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(buffers->d_velY, hVelY, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(buffers->d_velZ, hVelZ, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(buffers->d_masses, hMasses, bytes, cudaMemcpyHostToDevice));
    return NBODY_CUDA_SUCCESS;
}

extern "C" NBodyStatus_int nbCUDABuffersDownloadBodies(const struct NBodyCUDABuffers* buffers,
                                                       double* hPosX,
                                                       double* hPosY,
                                                       double* hPosZ,
                                                       double* hVelX,
                                                       double* hVelY,
                                                       double* hVelZ,
                                                       int nbody)
{
    if (!buffers || nbody != buffers->nbody) return NBODY_CUDA_ERROR;
    const size_t bytes = (size_t) nbody * sizeof(double);

    CUDA_CHECK(cudaMemcpy(hPosX, buffers->d_posX, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hPosY, buffers->d_posY, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hPosZ, buffers->d_posZ, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hVelX, buffers->d_velX, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hVelY, buffers->d_velY, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hVelZ, buffers->d_velZ, bytes, cudaMemcpyDeviceToHost));
    return NBODY_CUDA_SUCCESS;
}

/* opt #8 elaborate: kick off async D2H of bodies into the buffers'
 * pinned host buffer (h_pinnedBodies) on bestLikeStream. Records
 * bestLikeEvent on completion. Caller must NOT touch h_pinnedBodies
 * until either nbCUDABuffersWaitAsyncBodies has returned or the
 * caller has cudaEventSynchronize'd bestLikeEvent itself.
 *
 * The bestLikeStream first waits on the default stream (via a
 * temporary event) so the D2H sees the just-completed step's
 * outputs, then queues 6 cudaMemcpyAsync calls — one per SoA axis. */
extern "C" NBodyStatus_int nbCUDABuffersStartAsyncBodyMarshal(struct NBodyCUDABuffers* buffers)
{
    if (!buffers || !buffers->h_pinnedBodies || !buffers->bestLikeStream) return NBODY_CUDA_ERROR;
    if (buffers->bestLikePending) return NBODY_CUDA_ERROR;

    /* Make bestLikeStream wait for the default stream so we capture
     * the most-recent step's body data, not last step's. */
    cudaEvent_t deps;
    if (cudaEventCreateWithFlags(&deps, cudaEventDisableTiming) != cudaSuccess) return NBODY_CUDA_ERROR;
    if (cudaEventRecord(deps, 0) != cudaSuccess) { cudaEventDestroy(deps); return NBODY_CUDA_ERROR; }
    if (cudaStreamWaitEvent(buffers->bestLikeStream, deps, 0) != cudaSuccess) {
        cudaEventDestroy(deps); return NBODY_CUDA_ERROR;
    }
    cudaEventDestroy(deps);

    const size_t bytes = (size_t) buffers->nbody * sizeof(double);
    double* p = buffers->h_pinnedBodies;
    cudaStream_t s = buffers->bestLikeStream;

    if (cudaMemcpyAsync(p + 0 * buffers->nbody, buffers->d_posX, bytes, cudaMemcpyDeviceToHost, s) != cudaSuccess) return NBODY_CUDA_ERROR;
    if (cudaMemcpyAsync(p + 1 * buffers->nbody, buffers->d_posY, bytes, cudaMemcpyDeviceToHost, s) != cudaSuccess) return NBODY_CUDA_ERROR;
    if (cudaMemcpyAsync(p + 2 * buffers->nbody, buffers->d_posZ, bytes, cudaMemcpyDeviceToHost, s) != cudaSuccess) return NBODY_CUDA_ERROR;
    if (cudaMemcpyAsync(p + 3 * buffers->nbody, buffers->d_velX, bytes, cudaMemcpyDeviceToHost, s) != cudaSuccess) return NBODY_CUDA_ERROR;
    if (cudaMemcpyAsync(p + 4 * buffers->nbody, buffers->d_velY, bytes, cudaMemcpyDeviceToHost, s) != cudaSuccess) return NBODY_CUDA_ERROR;
    if (cudaMemcpyAsync(p + 5 * buffers->nbody, buffers->d_velZ, bytes, cudaMemcpyDeviceToHost, s) != cudaSuccess) return NBODY_CUDA_ERROR;
    if (cudaEventRecord(buffers->bestLikeEvent, s) != cudaSuccess) return NBODY_CUDA_ERROR;

    /* Make the default stream wait on bestLikeEvent so the NEXT
     * step's integration kernel (which writes d_posX/Y/Z and
     * d_velX/Y/Z) can't run until the async D2H finishes reading
     * them. Without this, step N+1's integration races with step
     * N's async D2H. The host can still proceed in parallel —
     * we're only enforcing GPU-stream ordering. */
    if (cudaStreamWaitEvent(0, buffers->bestLikeEvent, 0) != cudaSuccess) return NBODY_CUDA_ERROR;

    buffers->bestLikePending = 1;
    return NBODY_CUDA_SUCCESS;
}

/* opt #8 elaborate: block until the in-flight async D2H has finished,
 * then copy the pinned SoA into the caller's per-axis pointers.
 * Pointers may alias the buffer's pinned region — we use plain
 * memcpy. Resets bestLikePending. */
extern "C" NBodyStatus_int nbCUDABuffersWaitAsyncBodies(struct NBodyCUDABuffers* buffers,
                                                        double* hPosX, double* hPosY, double* hPosZ,
                                                        double* hVelX, double* hVelY, double* hVelZ)
{
    if (!buffers || !buffers->bestLikeEvent) return NBODY_CUDA_ERROR;
    if (!buffers->bestLikePending) return NBODY_CUDA_ERROR;

    if (cudaEventSynchronize(buffers->bestLikeEvent) != cudaSuccess) return NBODY_CUDA_ERROR;

    const size_t bytes = (size_t) buffers->nbody * sizeof(double);
    const double* p = buffers->h_pinnedBodies;
    if (hPosX) memcpy(hPosX, p + 0 * buffers->nbody, bytes);
    if (hPosY) memcpy(hPosY, p + 1 * buffers->nbody, bytes);
    if (hPosZ) memcpy(hPosZ, p + 2 * buffers->nbody, bytes);
    if (hVelX) memcpy(hVelX, p + 3 * buffers->nbody, bytes);
    if (hVelY) memcpy(hVelY, p + 4 * buffers->nbody, bytes);
    if (hVelZ) memcpy(hVelZ, p + 5 * buffers->nbody, bytes);

    buffers->bestLikePending = 0;
    return NBODY_CUDA_SUCCESS;
}

extern "C" int nbCUDABuffersIsAsyncMarshalPending(const struct NBodyCUDABuffers* buffers)
{
    return (buffers && buffers->bestLikePending) ? 1 : 0;
}

extern "C" NBodyStatus_int nbCUDABuffersUploadAccels(struct NBodyCUDABuffers* buffers,
                                                     const double* hAccX,
                                                     const double* hAccY,
                                                     const double* hAccZ,
                                                     int nbody)
{
    if (!buffers || nbody != buffers->nbody) return NBODY_CUDA_ERROR;
    const size_t bytes = (size_t) nbody * sizeof(double);

    CUDA_CHECK(cudaMemcpy(buffers->d_accX, hAccX, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(buffers->d_accY, hAccY, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(buffers->d_accZ, hAccZ, bytes, cudaMemcpyHostToDevice));
    return NBODY_CUDA_SUCCESS;
}

extern "C" NBodyStatus_int nbCUDABuffersDownloadAccels(const struct NBodyCUDABuffers* buffers,
                                                       double* hAccX,
                                                       double* hAccY,
                                                       double* hAccZ,
                                                       int nbody)
{
    if (!buffers || nbody != buffers->nbody) return NBODY_CUDA_ERROR;
    const size_t bytes = (size_t) nbody * sizeof(double);

    CUDA_CHECK(cudaMemcpy(hAccX, buffers->d_accX, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hAccY, buffers->d_accY, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hAccZ, buffers->d_accZ, bytes, cudaMemcpyDeviceToHost));
    return NBODY_CUDA_SUCCESS;
}

/* Block size for per-body kernels. 256 is a safe default across the
 * target architectures (V100 sm_70 / A100 sm_80 / Orin sm_87): high
 * occupancy, divisible by warp size (32), and well under the 1024
 * thread-per-block limit. Override at compile time via
 * -DNBODY_CUDA_BLOCK=N if profiling on a specific arch warrants it. */
#ifndef NBODY_CUDA_BLOCK
  #define NBODY_CUDA_BLOCK 256
#endif

/* ----- Phase 3: direct N-squared gravity force kernel ----- */

/* Tile-based all-pairs gravity. One thread per body i; each block
 * cooperatively pre-loads BLOCK bodies' positions+masses into shared
 * memory, then each thread accumulates contributions from those
 * BLOCK source bodies. Repeats over (nbody / BLOCK) tiles.
 *
 * Mirrors nbody_kernels.cl:2257-2362 (forceCalculation_Exact) but
 * without the external Milky Way / LMC potential, which will be
 * added in a follow-up phase. The arithmetic per pair is identical:
 *   dr     = src - dst
 *   rSq    = dr.dr + EPS2
 *   r      = sqrt(rSq)
 *   ai     = mass / (r * rSq)         <- Newton: G*m / r^3 (G folded
 *                                         into mass units upstream)
 *   acc   += ai * dr
 *
 * Note: unlike OpenCL we don't require nbody to be a multiple of
 * BLOCK. The tile loop runs ceil(nbody/BLOCK) times and per-thread
 * loads in the inner loop bounds-check via a runtime guard. The
 * branch is predicated and outside the FP-heavy inner work, so the
 * cost is negligible. Padding-style optimization can come back as
 * part of Phase 6 if profiling motivates it. */
__global__ void nbCUDAForceExactKernel(const double* __restrict__ d_posX,
                                       const double* __restrict__ d_posY,
                                       const double* __restrict__ d_posZ,
                                       const double* __restrict__ d_masses,
                                       double* __restrict__ d_accX,
                                       double* __restrict__ d_accY,
                                       double* __restrict__ d_accZ,
                                       const int    nbody,
                                       const double eps2)
{
    __shared__ double sxs[NBODY_CUDA_BLOCK];
    __shared__ double sys[NBODY_CUDA_BLOCK];
    __shared__ double szs[NBODY_CUDA_BLOCK];
    __shared__ double sms[NBODY_CUDA_BLOCK];

    const int tid  = threadIdx.x;
    const int gtid = blockIdx.x * blockDim.x + tid;

    /* Each thread targets one body i. Threads beyond nbody just
     * participate in cooperative loads (need full block to fill
     * shared memory) but skip the final write. */
    const bool active = gtid < nbody;
    const double px = active ? d_posX[gtid] : 0.0;
    const double py = active ? d_posY[gtid] : 0.0;
    const double pz = active ? d_posZ[gtid] : 0.0;

    double ax = 0.0;
    double ay = 0.0;
    double az = 0.0;

    const int nTiles = (nbody + NBODY_CUDA_BLOCK - 1) / NBODY_CUDA_BLOCK;

    for (int t = 0; t < nTiles; ++t)
    {
        const int srcIdx = t * NBODY_CUDA_BLOCK + tid;

        /* Cooperative load: each thread fetches one source body into
         * shared memory. Threads loading past nbody fill with mass=0
         * so contributions from the padding are exactly zero — the
         * reciprocal-distance scalar gets multiplied by zero and the
         * accumulator is unaffected by self-interaction at i==j (the
         * difference vector is zero, so ai*dr is zero too). */
        if (srcIdx < nbody)
        {
            sxs[tid] = d_posX[srcIdx];
            sys[tid] = d_posY[srcIdx];
            szs[tid] = d_posZ[srcIdx];
            sms[tid] = d_masses[srcIdx];
        }
        else
        {
            sxs[tid] = 0.0;
            sys[tid] = 0.0;
            szs[tid] = 0.0;
            sms[tid] = 0.0;
        }
        __syncthreads();

        /* Pairwise inner loop. The compiler unrolls this when the loop
         * bound is a compile-time constant (NBODY_CUDA_BLOCK). */
        #pragma unroll 8
        for (int k = 0; k < NBODY_CUDA_BLOCK; ++k)
        {
            const double dx = sxs[k] - px;
            const double dy = sys[k] - py;
            const double dz = szs[k] - pz;

            /* CPU bit-match: separate ops, NOT fma. CPU computes
             *   drSq = mw_sqrv(dr) + eps2 (separate mul+add per axis,
             *     since `-ffp-contract=off` blocks FMA fusion)
             *   drab = sqrt(drSq); phii = mass/drab; mor3 = phii/drSq
             *   acc.x += mor3 * dr.x  (separate mul+add, no FMA)
             * The OpenCL kernel used `mad()` (fused) and `1 mul + 1 div`
             * for rounding speed at the cost of CPU bit-equality. */
            const double rSq  = (dx*dx + dy*dy + dz*dz) + eps2;
            const double r    = sqrt(rSq);
            const double phii = sms[k] / r;
            const double ai   = phii / rSq;

            ax += ai * dx;
            ay += ai * dy;
            az += ai * dz;
        }
        __syncthreads();
    }

    if (active)
    {
        d_accX[gtid] = ax;
        d_accY[gtid] = ay;
        d_accZ[gtid] = az;
    }
}

extern "C" NBodyStatus_int nbCUDALaunchForceExact(struct NBodyCUDABuffers* buffers,
                                                  int nbody,
                                                  double eps2)
{
    if (!buffers || nbody != buffers->nbody || nbody <= 0)
    {
        return NBODY_CUDA_ERROR;
    }

    const int block = NBODY_CUDA_BLOCK;
    const int grid  = (nbody + block - 1) / block;

    nbCUDAForceExactKernel<<<grid, block>>>(buffers->d_posX,
                                            buffers->d_posY,
                                            buffers->d_posZ,
                                            buffers->d_masses,
                                            buffers->d_accX,
                                            buffers->d_accY,
                                            buffers->d_accZ,
                                            nbody,
                                            eps2);

    cudaError_t launchErr = cudaGetLastError();
    if (launchErr != cudaSuccess)
    {
        fprintf(stderr, "[nbody_cuda] forceExact kernel launch failed: %s\n",
                cudaGetErrorString(launchErr));
        return NBODY_CUDA_ERROR;
    }
    return NBODY_CUDA_SUCCESS;
}

/* ----- Phase 4: Barnes-Hut tree construction kernels -----
 *
 * The tree is stored across the body+cell SoA arrays:
 *   indices [0, nbody)         hold real bodies (positions, masses)
 *   indices [nbody, nNode]     hold tree cells (synthesized centroids)
 *   index   nNode              is the root cell
 * Cell child pointers live in d_child[NSUB*node + slot]; cells use
 * d_count/d_start for body counts and the start index in the sorted
 * permutation. _treeStatus->bottom shrinks downward as cells are
 * allocated during buildTree.
 *
 * Translation table from the OpenCL kernels in
 * nbody/kernels/nbody_kernels.cl:
 *    __kernel             -> __global__
 *    __local              -> __shared__
 *    __local volatile     -> __shared__ volatile
 *    barrier(CLK_LMF)     -> __syncthreads()
 *    mem_fence(CLK_GMF)   -> __threadfence()
 *    atom_inc / atom_cas  -> atomicAdd(.., 1) / atomicCAS
 *    mad / fmin / fmax    -> fma / fmin / fmax (FP64; CUDA has all)
 */

/* boundingBox: parallel reduction over all body positions to find the
 * AABB; the last block to finish writes the root cell into node nNode
 * and resets the tree counters in d_treeStatus. Mirrors
 * nbody_kernels.cl:736-857. */
__global__ void nbCUDABoundingBoxKernel(const double* __restrict__ d_posX,
                                        const double* __restrict__ d_posY,
                                        const double* __restrict__ d_posZ,
                                        double* __restrict__ d_minX,
                                        double* __restrict__ d_minY,
                                        double* __restrict__ d_minZ,
                                        double* __restrict__ d_maxX,
                                        double* __restrict__ d_maxY,
                                        double* __restrict__ d_maxZ,
                                        double* __restrict__ d_critRadii,
                                        double* __restrict__ d_mass,
                                        double* __restrict__ d_posXOut,
                                        double* __restrict__ d_posYOut,
                                        double* __restrict__ d_posZOut,
                                        int* __restrict__ d_start,
                                        int* __restrict__ d_child,
                                        struct NBodyCUDATreeStatus* d_treeStatus,
                                        int nbody,
                                        int nNode)
{
    /* Match CPU's expandBox semantics:
     *   - Root cell is at the origin (0,0,0), NOT the body bbox midpoint.
     *   - Compute xyzmax = max(|body_pos - origin|) across all bodies/axes.
     *   - rsize starts at ctx->treeRSize and doubles until rsize >= 2*xyzmax.
     *   - rsize PERSISTS across steps and only grows (we mirror this by
     *     reading the previous step's d_treeStatus->radius as the initial
     *     value; on the very first step, init logic seeds it to the
     *     ctx->treeRSize value via d_treeStatus->radius before launch).
     * sMaxX/Y/Z[] now hold per-thread max(|body_axis|) instead of min/max
     * range. */
    __shared__ volatile double sMaxX[NBODY_CUDA_BLOCK];
    __shared__ volatile double sMaxY[NBODY_CUDA_BLOCK];
    __shared__ volatile double sMaxZ[NBODY_CUDA_BLOCK];

    const unsigned int tid = threadIdx.x;
    sMaxX[tid] = 0.0;
    sMaxY[tid] = 0.0;
    sMaxZ[tid] = 0.0;
    __syncthreads();

    const unsigned int inc = blockDim.x * gridDim.x;
    unsigned int j = tid + blockIdx.x * blockDim.x;

    /* Per-thread scan: accumulate per-axis fabs max. */
    while ((int) j < nbody)
    {
        sMaxX[tid] = fmax(sMaxX[tid], fabs(d_posX[j]));
        sMaxY[tid] = fmax(sMaxY[tid], fabs(d_posY[j]));
        sMaxZ[tid] = fmax(sMaxZ[tid], fabs(d_posZ[j]));
        j += inc;
    }

    /* Tree reduction in shared memory. */
    j = blockDim.x >> 1;
    while (j > 0)
    {
        __syncthreads();
        if (tid < j)
        {
            sMaxX[tid] = fmax(sMaxX[tid], sMaxX[tid + j]);
            sMaxY[tid] = fmax(sMaxY[tid], sMaxY[tid + j]);
            sMaxZ[tid] = fmax(sMaxZ[tid], sMaxZ[tid + j]);
        }
        j >>= 1;
    }

    if (tid == 0)
    {
        /* Stash this block's per-axis maxes (reuse d_max* slots). */
        const unsigned int g = blockIdx.x;
        d_maxX[g] = sMaxX[0];
        d_maxY[g] = sMaxY[0];
        d_maxZ[g] = sMaxZ[0];
        /* Unused now but keep min slots zeroed so a future-min reduce
         * on stale data doesn't pollute downstream debug. */
        d_minX[g] = 0.0; d_minY[g] = 0.0; d_minZ[g] = 0.0;
        __threadfence();

        /* Last block to arrive folds the per-block maxes and snaps
         * radius to power-of-2 doubling (matches CPU expandBox). */
        const int last = gridDim.x - 1;
        if (atomicAdd(&d_treeStatus->blkCnt, 1) == last)
        {
            double mxX = sMaxX[0], mxY = sMaxY[0], mxZ = sMaxZ[0];
            for (int g2 = 0; g2 <= last; ++g2)
            {
                mxX = fmax(mxX, d_maxX[g2]);
                mxY = fmax(mxY, d_maxY[g2]);
                mxZ = fmax(mxZ, d_maxZ[g2]);
            }
            const double xyzmax = fmax(fmax(mxX, mxY), mxZ);

            /* Persist & grow: start from the previous step's radius
             * (or the initial seed) and double until radius >= 2*xyzmax.
             * CPU's t->rsize only ever grows; we mirror that.
             *
             * On the very first call, d_treeStatus->radius is 0 (cudaMalloc).
             * Seed to 4.0 to match CPU's DEFAULT_TREE_ROOT_SIZE. The lua
             * default doesn't override this; if a future workload uses a
             * non-default treeRSize, this should be plumbed through as
             * a kernel parameter. */
            double radius = d_treeStatus->radius;
            if (!(radius > 0.0)) radius = 4.0;
            while (radius < 2.0 * xyzmax)
            {
                radius *= 2.0;
            }

            d_treeStatus->radius  = radius;
            d_treeStatus->bottom  = nNode;
            d_treeStatus->blkCnt  = 0;   /* Reset for next step */
            d_treeStatus->doneCnt = 0;

            if (d_critRadii)              /* present for TREECODE/SW93 */
            {
                d_critRadii[nNode] = radius;
            }

            /* Initialize root cell at the ORIGIN (0,0,0), matching
             * CPU's mw_zerov(Pos(t->root)). */
            d_mass[nNode]    = -1.0;
            d_start[nNode]   = 0;
            d_posXOut[nNode] = 0.0;
            d_posYOut[nNode] = 0.0;
            d_posZOut[nNode] = 0.0;

            #pragma unroll
            for (int k = 0; k < NBODY_CUDA_NSUB; ++k)
            {
                d_child[NBODY_CUDA_NSUB * nNode + k] = -1;
            }
        }
    }
}

extern "C" NBodyStatus_int nbCUDALaunchBoundingBox(struct NBodyCUDABuffers* buffers,
                                                   int nbody,
                                                   int nNode)
{
    if (!buffers || !buffers->d_treeStatus) return NBODY_CUDA_ERROR;
    /* Use one block per SM so each block's per-block reduction has a
     * meaningful chunk of work. Matches the OpenCL ws->blocks[0] sizing
     * which equals di->maxCompUnits. */
    const int block = NBODY_CUDA_BLOCK;
    const int grid  = buffers->numSMs > 0 ? buffers->numSMs : 1;

    nbCUDABoundingBoxKernel<<<grid, block>>>(buffers->d_posX,
                                             buffers->d_posY,
                                             buffers->d_posZ,
                                             buffers->d_minX,
                                             buffers->d_minY,
                                             buffers->d_minZ,
                                             buffers->d_maxX,
                                             buffers->d_maxY,
                                             buffers->d_maxZ,
                                             buffers->d_critRadii,
                                             buffers->d_masses,
                                             buffers->d_posX,
                                             buffers->d_posY,
                                             buffers->d_posZ,
                                             buffers->d_start,
                                             buffers->d_child,
                                             buffers->d_treeStatus,
                                             nbody,
                                             nNode);
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        fprintf(stderr, "[nbody_cuda] boundingBox launch: %s\n", cudaGetErrorString(e));
        return NBODY_CUDA_ERROR;
    }
    return NBODY_CUDA_SUCCESS;
}

/* buildTreeClear: blank the cell-side child pointers ahead of buildTree.
 * Mirrors nbody_kernels.cl:879-896. */
#define NBODY_CUDA_NULL_BODY (-1)
#define NBODY_CUDA_WARPSIZE  32

__global__ void nbCUDABuildTreeClearKernel(int* __restrict__ d_child,
                                           double* __restrict__ d_posX,
                                           double* __restrict__ d_posY,
                                           double* __restrict__ d_posZ,
                                           int nbody,
                                           int nNode)
{
    const int top = NBODY_CUDA_NSUB * nNode;
    const int bot = NBODY_CUDA_NSUB * nbody;
    const int inc = blockDim.x * gridDim.x;
    int k = (bot & (-NBODY_CUDA_WARPSIZE)) + (int) (blockIdx.x * blockDim.x + threadIdx.x);
    if (k < bot) k += inc;
    while (k < top)
    {
        d_child[k] = NBODY_CUDA_NULL_BODY;
        k += inc;
    }

}

extern "C" NBodyStatus_int nbCUDALaunchBuildTreeClear(struct NBodyCUDABuffers* buffers,
                                                      int nNode)
{
    if (!buffers || !buffers->d_child) return NBODY_CUDA_ERROR;
    const int block = NBODY_CUDA_BLOCK;
    const int grid  = buffers->numSMs > 0 ? buffers->numSMs : 1;
    nbCUDABuildTreeClearKernel<<<grid, block>>>(buffers->d_child,
                                                buffers->d_posX,
                                                buffers->d_posY,
                                                buffers->d_posZ,
                                                buffers->nbody, nNode);
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        fprintf(stderr, "[nbody_cuda] buildTreeClear launch: %s\n", cudaGetErrorString(e));
        return NBODY_CUDA_ERROR;
    }
    return NBODY_CUDA_SUCCESS;
}

/* ===========================================================
 * Morton-code based deterministic parallel buildTree.
 * ===========================================================
 *
 * The default atomicCAS-based parallel buildTree above produces a
 * GEOMETRICALLY-CORRECT tree across runs but its cell INDEX assignment
 * is timing-dependent (atomicSub on d_treeStatus->bottom returns
 * indices in CAS-winner order). With pinned host memory in play (opt #8
 * elaborate), the CUDA runtime's scheduling perturbation can flip
 * resolution of multi-way races, producing different INTERMEDIATE
 * cell-allocation orders. While the geometric tree is unchanged, the
 * scheduling perturbation also alters warp execution timing in
 * downstream kernels (forceTree warp-vote, summarization poll order),
 * which combined with FP non-associativity yields bit-different
 * per-step results.
 *
 * The Morton-code path eliminates atomicCAS entirely: it computes a
 * 63-bit Morton (z-order) code per body, deterministically sorts
 * bodies by (Morton, body_index), and constructs the tree top-down by
 * partitioning the sorted body list at each octree level. No atomic
 * contention, fully deterministic cell index assignment regardless of
 * GPU scheduling.
 *
 * Output is in the SAME format as the atomicCAS kernel:
 *   d_child[NSUB*cell + octant] = body_idx (0..nbody-1)
 *                                OR cell_idx (nbody..nNode)
 *                                OR -1 (empty)
 *   d_posX/Y/Z[cell] = cell geometric centre
 *   d_critRadii[cell] = cell size (passes through summarization
 *                       which overwrites with the actual opening
 *                       criterion radius)
 *   d_treeStatus->bottom = lowest cell index used
 *   d_treeStatus->radius = root size (unchanged from bounding box pass)
 */

/* Morton128 + Morton128Less + Morton128Decomposer are defined near
 * the top of this TU (above NBodyCUDABuffers) because the CUB
 * temp-storage query in nbCUDATreeBuffersAlloc needs the complete
 * type. The kernels below just use those declarations. */

/* Expand 42-bit input to 126 output bits (3*i positions). Returns
 * the result as a (lo, hi) pair packed into 128 bits. Linear loop
 * is fine — called once per body per step, parallel across 40K
 * threads, so per-build cost is in the µs range. */
__device__ __forceinline__ void mortonExpandBits42(unsigned long long v,
                                                    unsigned long long* out_lo,
                                                    unsigned long long* out_hi)
{
    unsigned long long lo = 0, hi = 0;
    #pragma unroll
    for (int i = 0; i < 42; ++i)
    {
        unsigned long long bit = (v >> i) & 1ULL;
        int pos = 3 * i;
        if (pos < 64) lo |= bit << pos;
        else          hi |= bit << (pos - 64);
    }
    *out_lo = lo;
    *out_hi = hi;
}

/* Extract the 3-bit octant code at tree depth d from a 128-bit
 * Morton key. Octant bits live at positions (125 - 3d, 124 - 3d,
 * 123 - 3d) of the 126-bit Morton; we read the 3-bit slice starting
 * at p = 123 - 3d.
 *
 * For 42-bit-per-axis Morton, depth ranges over [0, 41] before the
 * Morton bits run out. The fused tree builder caps the level loop
 * at NBODY_CUDA_MAXDEPTH = 26, well within this range. */
__device__ __forceinline__ unsigned int morton128_octant(const Morton128& m, int depth)
{
    const int p = 123 - 3 * depth;
    if (p >= 64)
    {
        return (unsigned int) ((m.hi >> (p - 64)) & 7ULL);
    }
    if (p <= 61)
    {
        return (unsigned int) ((m.lo >> p) & 7ULL);
    }
    /* Straddle: bits at p=62 or p=63 straddle the 64-bit boundary. */
    if (p == 63)
    {
        /* bits 63 (lo), 64 (hi[0]), 65 (hi[1]) */
        unsigned long long low_part = (m.lo >> 63) & 1ULL;            /* bit 0 of result */
        unsigned long long hi_part  = (m.hi & 3ULL) << 1;             /* bits 1..2 of result */
        return (unsigned int) (low_part | hi_part);
    }
    /* p == 62: bits 62 (lo), 63 (lo), 64 (hi[0]) */
    unsigned long long low_part = (m.lo >> 62) & 3ULL;                /* bits 0..1 of result */
    unsigned long long hi_part  = (m.hi & 1ULL) << 2;                 /* bit 2 of result */
    return (unsigned int) (low_part | hi_part);
}

/* Compute 128-bit Morton code for each body. 42 bits per axis × 3
 * axes = 126 output bits. The code packs interleaved bits of
 * normalised (px, py, pz) ∈ [0,1] so that bodies in the same root
 * octant share a 3-bit prefix, same depth-2 octant share 6-bit prefix,
 * etc. Octant code at depth d = (X<<2)|(Y<<1)|Z, matching the
 * per-cell octant index used by the BH walk and summarization.
 *
 * Normalisation: pos ∈ [-rsize/2, +rsize/2] → t = (pos+half)/rsize
 * ∈ [0,1]. Multiply by 2^42 and truncate. 42 bits / axis is enough
 * to resolve any cell down to legacy's MAXDEPTH=26 (which only
 * resolves bodies separated by 2^-26 of bbox). Beyond depth 26
 * Morton still has 16 bits of headroom.
 *
 * Bit interpretation: at depth d (0 = root), the octant index is
 * bits (125-3d, 124-3d, 123-3d) of the 126-bit Morton. The
 * morton128_octant() helper extracts this 3-bit slice from a
 * (lo, hi) pair. */
__global__ void nbCUDAComputeMortonKernel(
    const double* __restrict__ d_posX,
    const double* __restrict__ d_posY,
    const double* __restrict__ d_posZ,
    Morton128* __restrict__ d_morton,
    int* __restrict__ d_idx,
    const struct NBodyCUDATreeStatus* __restrict__ d_treeStatus,
    int nbody)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nbody) return;

    /* Root cell SIZE = d_treeStatus->radius (full edge length; bodies
     * live in [-rsize/2, +rsize/2]). Read once per kernel; the value
     * was set by boundingBox earlier in this step. */
    const double rsize = d_treeStatus->radius;

    const double px = d_posX[i];
    const double py = d_posY[i];
    const double pz = d_posZ[i];

    /* Map pos in [-rsize/2, +rsize/2] -> [0, 1] -> [0, 1<<42). Use
     * scale = 2^42 so that t = 0.5 (body at centre) lands on q = 2^41,
     * with bit-41 set — matches the atomicCAS kernel's "<=" octant
     * encoding (centre body goes to high octant). For t = 1.0 (body
     * at +rsize/2), q = 2^42, clamped to 2^42 - 1, still in high
     * octant. */
    const double scale = (double) (1ULL << 42);
    const double half = 0.5 * rsize;
    double tx = (px + half) / rsize;
    double ty = (py + half) / rsize;
    double tz = (pz + half) / rsize;
    if (tx < 0.0) tx = 0.0; else if (tx > 1.0) tx = 1.0;
    if (ty < 0.0) ty = 0.0; else if (ty > 1.0) ty = 1.0;
    if (tz < 0.0) tz = 0.0; else if (tz > 1.0) tz = 1.0;
    unsigned long long qx = (unsigned long long) (tx * scale);
    unsigned long long qy = (unsigned long long) (ty * scale);
    unsigned long long qz = (unsigned long long) (tz * scale);
    const unsigned long long MAX_Q = (1ULL << 42) - 1ULL;
    if (qx > MAX_Q) qx = MAX_Q;
    if (qy > MAX_Q) qy = MAX_Q;
    if (qz > MAX_Q) qz = MAX_Q;

    /* Expand each axis to a 126-bit (lo, hi) pair, then combine via
     * (mx<<2) | (my<<1) | mz across the 128-bit pair. */
    unsigned long long mx_lo, mx_hi, my_lo, my_hi, mz_lo, mz_hi;
    mortonExpandBits42(qx, &mx_lo, &mx_hi);
    mortonExpandBits42(qy, &my_lo, &my_hi);
    mortonExpandBits42(qz, &mz_lo, &mz_hi);

    /* 128-bit left shift by 2: lo<<2, hi = (hi<<2) | (lo>>62). */
    const unsigned long long mxs_lo = mx_lo << 2;
    const unsigned long long mxs_hi = (mx_hi << 2) | (mx_lo >> 62);
    /* 128-bit left shift by 1: lo<<1, hi = (hi<<1) | (lo>>63). */
    const unsigned long long mys_lo = my_lo << 1;
    const unsigned long long mys_hi = (my_hi << 1) | (my_lo >> 63);
    /* mz already at shift 0. */

    Morton128 m;
    m.lo = mxs_lo | mys_lo | mz_lo;
    m.hi = mxs_hi | mys_hi | mz_hi;
    d_morton[i] = m;
    d_idx[i] = i;
}

/* The previous launch-wrapper entry points
 * (nbCUDALaunchComputeMorton, nbCUDALaunchSortByMorton) have been
 * removed; the Morton orchestrator below calls
 * nbCUDAComputeMortonKernel and thrust::sort_by_key directly. */

/* Fused per-level kernel: count octants AND emit children in one
 * pass. Cell indices are allocated via atomicSub on d_treeStatus->bottom
 * — non-deterministic across runs, but the tree TOPOLOGY (which
 * bodies live under which cell) is deterministic from sorted-Morton
 * partitioning, which is what matters for downstream FP determinism:
 *
 *   - Summarization (line 2380) COMPACTS non-empty children into
 *     slots [0..n) in octant-iteration order, so its 8-child sum is
 *     deterministic from tree topology alone.
 *   - Force walk visits cells via d_child pointers; visit ORDER is
 *     determined by tree topology, not by cell indices.
 *
 * Thus atomicSub-allocated cells give bit-identical likelihoods
 * across runs even though their integer indices differ.
 *
 * Each thread handles one cell from the level-in queue. New cells go
 * into the level-out queue at positions allocated via atomicAdd on
 * d_outCount. d_inCount/d_outCount are pure device-side counters: no
 * host syncs needed between levels. */
__global__ void nbCUDAMortonFusedKernel(
    const Morton128* __restrict__ d_morton,
    const int* __restrict__ d_sortedIdx,
    const int* __restrict__ d_lvlInCells,
    const int* __restrict__ d_lvlInStart,
    const int* __restrict__ d_lvlInEnd,
    const int* __restrict__ d_lvlInDepth,
    const int* __restrict__ d_inCount,
    int* __restrict__ d_lvlOutCells,
    int* __restrict__ d_lvlOutStart,
    int* __restrict__ d_lvlOutEnd,
    int* __restrict__ d_lvlOutDepth,
    int* __restrict__ d_outCount,
    int* __restrict__ d_child,
    double* __restrict__ d_posX,
    double* __restrict__ d_posY,
    double* __restrict__ d_posZ,
    double* __restrict__ d_critRadii,
    struct NBodyCUDATreeStatus* d_treeStatus,
    int nbody, int nNode)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    const int inCount = *d_inCount;
    if (c >= inCount) return;

    const int cellIdx = d_lvlInCells[c];
    const int start   = d_lvlInStart[c];
    const int end     = d_lvlInEnd[c];
    const int depth   = d_lvlInDepth[c];

    /* Per-cell depth tracking: bubble up to d_treeStatus->maxDepth so
     * downstream kernels (quadMoments) see a consistent max-depth
     * value. Each cell's `depth + 1` is the deepest LEVEL its
     * children populate at; threads racing on atomicMax is fine. */
    atomicMax(&d_treeStatus->maxDepth, depth + 1);

    /* 128-bit Morton boundary search: 7 binary searches in the
     * sorted d_morton range, using morton128_octant() to extract the
     * 3-bit octant code at this depth. 42 bits/axis covers depth
     * 0..41 — well past the legacy MAXDEPTH=26 cap, so the
     * rank-partition fallback below should never fire on a realistic
     * WU. */
    int boundaries[9];
    boundaries[0] = start;
    boundaries[8] = end;
    if (depth <= 41)
    {
        #pragma unroll
        for (int o = 1; o < 8; ++o)
        {
            int lo = start, hi = end;
            while (lo < hi)
            {
                int mid = (lo + hi) >> 1;
                unsigned int oct = morton128_octant(d_morton[mid], depth);
                if (oct < (unsigned int) o) lo = mid + 1;
                else hi = mid;
            }
            boundaries[o] = lo;
        }
    }
    else
    {
        /* Defensive only: depth >41 should never happen for realistic
         * WUs given NBODY_CUDA_MAXDEPTH=26. If hit, partition the
         * sorted body range evenly by rank so the tree-build still
         * terminates without orphan cells (which would NaN-hang quad
         * moments). errorCode is set so the host knows precision
         * was exceeded. */
        const int rangeLen = end - start;
        #pragma unroll
        for (int o = 0; o <= 8; ++o)
        {
            boundaries[o] = start + (o * rangeLen) / 8;
        }
        if (threadIdx.x == 0 && blockIdx.x == 0)
            d_treeStatus->errorCode = 3;
    }

    /* Read parent cell geometry (set by boundingBox or by a previous
     * level's emit for non-root cells). */
    const double cx    = d_posX[cellIdx];
    const double cy    = d_posY[cellIdx];
    const double cz    = d_posZ[cellIdx];
    const double csize = d_critRadii[cellIdx];

    const double cellSize = 0.5  * csize;
    const double offset   = 0.25 * csize;

    #pragma unroll
    for (int o = 0; o < 8; ++o)
    {
        int s = boundaries[o];
        int e = boundaries[o + 1];
        int cnt = e - s;
        int slot = NBODY_CUDA_NSUB * cellIdx + o;

        if (cnt == 0)
        {
            d_child[slot] = -1;
        }
        else if (cnt == 1)
        {
            d_child[slot] = d_sortedIdx[s];
        }
        else if (depth + 1 > NBODY_CUDA_MAXDEPTH)
        {
            /* MAXDEPTH overflow — matches legacy atomicCAS's behavior:
             * legacy's split loop exits via `depth > NBODY_CUDA_MAXDEPTH`,
             * after which `d_child[NSUB*n+j] = i` places the body
             * currently being inserted, OVERWRITING the previously-placed
             * body in that slot. Net effect: one body retained per slot,
             * others lost; errorCode is set.
             *
             * The "i" body in legacy's race is determined by thread
             * scheduling (not deterministic across runs with pinned, but
             * deterministic without). To approximate legacy's choice, we
             * keep the body with the HIGHEST body_idx in the range — the
             * intuition is that legacy's stride-based body assignment
             * (thread tid handles bodies tid, tid+inc, tid+2*inc, ...)
             * tends to make higher-idx bodies "win" the deepest split
             * because they're processed in later iterations of each
             * thread's loop. This is an approximation, not bit-exact. */
            d_child[slot] = d_sortedIdx[e - 1];
            d_treeStatus->errorCode = 1;
        }
        else
        {
            int newCell = atomicSub(&d_treeStatus->bottom, 1) - 1;

            double cnx = cx + ((o & 4) ? offset : -offset);
            double cny = cy + ((o & 2) ? offset : -offset);
            double cnz = cz + ((o & 1) ? offset : -offset);

            d_posX[newCell] = cnx;
            d_posY[newCell] = cny;
            d_posZ[newCell] = cnz;
            d_critRadii[newCell] = cellSize;

            d_child[slot] = newCell;

            int q = atomicAdd(d_outCount, 1);
            d_lvlOutCells[q] = newCell;
            d_lvlOutStart[q] = s;
            d_lvlOutEnd[q]   = e;
            d_lvlOutDepth[q] = depth + 1;
        }
    }
}

/* Tiny kernel: in_count = out_count; out_count = 0. */
__global__ void nbCUDAMortonSwapCountersKernel(int* d_inCount, int* d_outCount)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        *d_inCount = *d_outCount;
        *d_outCount = 0;
    }
}

/* Tiny kernel: seed the level-in queue with the root cell AND set
 * counters (in_count=1, out_count=0). One thread does all of it. */
__global__ void nbCUDAMortonSeedRootKernel(int* d_lvlInCells,
                                           int* d_lvlInStart,
                                           int* d_lvlInEnd,
                                           int* d_lvlInDepth,
                                           int* d_inCount,
                                           int* d_outCount,
                                           int nNode,
                                           int nbody)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        d_lvlInCells[0] = nNode;
        d_lvlInStart[0] = 0;
        d_lvlInEnd[0]   = nbody;
        d_lvlInDepth[0] = 0;
        *d_inCount  = 1;
        *d_outCount = 0;
    }
}

/* ============================================================
 * nbCUDABuildTreeMorton — full deterministic tree builder.
 * ============================================================
 *
 * Phases:
 *   1. Compute Morton code per body and seed the sort permutation.
 *   2. thrust::sort_by_key on (morton, body_idx). Output sorted in
 *      ascending Morton order.
 *   3. Seed the level queue with the root cell covering [0, nbody)
 *      at depth 0.
 *   4. Loop: while inCount > 0 and depth <= MAXDEPTH:
 *      4a. Count kernel: per-cell, find 8 octant boundaries via
 *          binary search; set predicate[i] = 1 if octant i has >=2
 *          bodies.
 *      4b. thrust::exclusive_scan over predicate -> offsets.
 *      4c. Read offsets[N-1] + predicate[N-1] = totalNew via a
 *          pinned-host async readback.
 *      4d. Emit kernel: allocate (oldBottom-1, oldBottom-2, ...)
 *          deterministically, write d_child slots, set new cell
 *          geometry, enqueue children in out-queue.
 *      4e. oldBottom -= totalNew; swap in/out; inCount = totalNew.
 *   5. Write final bottom to d_treeStatus->bottom.
 *
 * The reads in 4c are the only host syncs. With max ~21 levels per
 * step and ~17000 steps per WU, that's ~360K sync points. Each sync
 * is ~10us, so ~3.6s of sync overhead per WU. Acceptable given the
 * determinism win.
 */
/* Per-phase timers, enabled by NBODY_BUILDTREE_MORTON_PROFILE=1. The
 * first NBODY_MORTON_PROFILE_STEPS calls write a CSV row to stderr
 * with elapsed microseconds per phase. */
#define NBODY_MORTON_PROFILE_STEPS 5

/* Issue the entire Morton tree-build pipeline onto `stream`. Used
 * both to capture the CUDA graph (first call) and as the fallback
 * if graph launch isn't available. All kernels are launched on
 * `stream` so they can be captured into a single graph. */
static cudaError_t nbCUDAMortonRecord(struct NBodyCUDABuffers* buffers,
                                      int nbody, int nNode,
                                      cudaStream_t stream)
{
    cudaError_t err;

    const int block = NBODY_CUDA_BLOCK;
    const int grid  = (nbody + block - 1) / block;

    /* Phase 1: Morton compute → *Scratch arrays. */
    nbCUDAComputeMortonKernel<<<grid, block, 0, stream>>>(
        buffers->d_posX, buffers->d_posY, buffers->d_posZ,
        buffers->d_mortonScratch, buffers->d_sortedIdxScratch,
        buffers->d_treeStatus, nbody);
    if ((err = cudaGetLastError()) != cudaSuccess) return err;

    /* Phase 2: CUB radix sort, decomposer-driven, on the same stream. */
    err = cub::DeviceRadixSort::SortPairs(
        buffers->d_sortTempStorage, buffers->sortTempBytes,
        (const Morton128*) buffers->d_mortonScratch,
        (Morton128*)       buffers->d_morton,
        (const int*)       buffers->d_sortedIdxScratch,
        (int*)             buffers->d_sortedIdx,
        nbody,
        Morton128Decomposer{},
        0, 128, stream);
    if (err != cudaSuccess) return err;

    /* Phase 3: seed root cell + reset counters. */
    nbCUDAMortonSeedRootKernel<<<1, 32, 0, stream>>>(
        buffers->d_lvlInCells, buffers->d_lvlInStart,
        buffers->d_lvlInEnd,   buffers->d_lvlInDepth,
        buffers->d_lvlInCount, buffers->d_lvlOutCount,
        nNode, nbody);
    if ((err = cudaGetLastError()) != cudaSuccess) return err;

    /* Phase 4: alternating fused + swap-counter, MAXDEPTH+1 iterations.
     * Pointers alternate per iteration so each iteration's `in` =
     * previous iteration's `out`. The pointer pairs cycle through
     * 2 values, so we just swap on host between launches. The kernel
     * args differ per iteration, baking into separate graph nodes. */
    int* lvlInCells   = buffers->d_lvlInCells;
    int* lvlInStart   = buffers->d_lvlInStart;
    int* lvlInEnd     = buffers->d_lvlInEnd;
    int* lvlInDepth   = buffers->d_lvlInDepth;
    int* lvlOutCells  = buffers->d_lvlOutCells;
    int* lvlOutStart  = buffers->d_lvlOutStart;
    int* lvlOutEnd    = buffers->d_lvlOutEnd;
    int* lvlOutDepth  = buffers->d_lvlOutDepth;
    for (int level = 0; level <= NBODY_CUDA_MAXDEPTH; ++level)
    {
        nbCUDAMortonFusedKernel<<<grid, block, 0, stream>>>(
            buffers->d_morton, buffers->d_sortedIdx,
            lvlInCells, lvlInStart, lvlInEnd, lvlInDepth,
            buffers->d_lvlInCount,
            lvlOutCells, lvlOutStart, lvlOutEnd, lvlOutDepth,
            buffers->d_lvlOutCount,
            buffers->d_child,
            buffers->d_posX, buffers->d_posY, buffers->d_posZ,
            buffers->d_critRadii,
            buffers->d_treeStatus,
            nbody, nNode);
        nbCUDAMortonSwapCountersKernel<<<1, 32, 0, stream>>>(
            buffers->d_lvlInCount, buffers->d_lvlOutCount);
        { int* t = lvlInCells; lvlInCells = lvlOutCells; lvlOutCells = t; }
        { int* t = lvlInStart; lvlInStart = lvlOutStart; lvlOutStart = t; }
        { int* t = lvlInEnd;   lvlInEnd   = lvlOutEnd;   lvlOutEnd   = t; }
        { int* t = lvlInDepth; lvlInDepth = lvlOutDepth; lvlOutDepth = t; }
    }
    return cudaGetLastError();
}

extern "C" NBodyStatus_int nbCUDABuildTreeMorton(struct NBodyCUDABuffers* buffers,
                                                 int nbody,
                                                 int nNode)
{
    if (!buffers || !buffers->d_treeStatus || !buffers->d_child) return NBODY_CUDA_ERROR;
    if (!buffers->d_morton || !buffers->d_sortedIdx) return NBODY_CUDA_ERROR;
    if (nbody != buffers->nbody) return NBODY_CUDA_ERROR;
    if (!buffers->mortonStream) return NBODY_CUDA_ERROR;

    static int profileCachedFlag = -1;
    static int profileStepCount = 0;
    if (profileCachedFlag < 0) {
        const char* e = getenv("NBODY_BUILDTREE_MORTON_PROFILE");
        profileCachedFlag = (e && e[0] && e[0] != '0') ? 1 : 0;
    }
    int doProfile = profileCachedFlag && (profileStepCount < NBODY_MORTON_PROFILE_STEPS);

    auto syncNow = [](){ cudaDeviceSynchronize(); };
    struct timespec t0, t1, t2, t3, t4, t5;
    if (doProfile) {
        syncNow();
        clock_gettime(CLOCK_MONOTONIC, &t0);
    }
    cudaError_t err;

    /* opt #24: capture-and-replay via CUDA graphs. First call captures
     * the entire Morton pipeline onto buffers->mortonStream and
     * instantiates an executable graph; subsequent calls replay the
     * graph (single launch instead of ~50 individual kernel launches).
     *
     * After launching (capture or replay), cudaStreamSynchronize the
     * mortonStream so the subsequent default-stream pipeline (Sort,
     * Summarization, etc.) sees the new tree. */
    if (!buffers->mortonGraphCaptured)
    {
        if ((err = cudaStreamBeginCapture(buffers->mortonStream,
                                          cudaStreamCaptureModeThreadLocal)) != cudaSuccess) {
            fprintf(stderr, "[nbody_cuda] morton graph beginCapture: %s\n", cudaGetErrorString(err));
            return NBODY_CUDA_ERROR;
        }
        if ((err = nbCUDAMortonRecord(buffers, nbody, nNode, buffers->mortonStream)) != cudaSuccess) {
            cudaStreamEndCapture(buffers->mortonStream, &buffers->mortonGraph);
            fprintf(stderr, "[nbody_cuda] morton record (capture): %s\n", cudaGetErrorString(err));
            return NBODY_CUDA_ERROR;
        }
        if ((err = cudaStreamEndCapture(buffers->mortonStream, &buffers->mortonGraph)) != cudaSuccess) {
            fprintf(stderr, "[nbody_cuda] morton graph endCapture: %s\n", cudaGetErrorString(err));
            return NBODY_CUDA_ERROR;
        }
        if ((err = cudaGraphInstantiate(&buffers->mortonGraphExec,
                                         buffers->mortonGraph,
                                         nullptr, nullptr, 0)) != cudaSuccess) {
            fprintf(stderr, "[nbody_cuda] morton graph instantiate: %s\n", cudaGetErrorString(err));
            return NBODY_CUDA_ERROR;
        }
        buffers->mortonGraphCaptured = 1;
    }

    if ((err = cudaGraphLaunch(buffers->mortonGraphExec, buffers->mortonStream)) != cudaSuccess) {
        fprintf(stderr, "[nbody_cuda] morton graph launch: %s\n", cudaGetErrorString(err));
        return NBODY_CUDA_ERROR;
    }
    if ((err = cudaStreamSynchronize(buffers->mortonStream)) != cudaSuccess) {
        fprintf(stderr, "[nbody_cuda] morton stream sync: %s\n", cudaGetErrorString(err));
        return NBODY_CUDA_ERROR;
    }
    if (doProfile) { syncNow(); clock_gettime(CLOCK_MONOTONIC, &t3); }
    /* Phases 1-4 were captured; t2 == t3 is fine for the legacy
     * profile output below. */
    t2 = t3;

    if (doProfile) {
        syncNow();
        clock_gettime(CLOCK_MONOTONIC, &t5);
        long us_morton     = (t2.tv_sec - t0.tv_sec)*1000000L + (t2.tv_nsec - t0.tv_nsec)/1000L;
        long us_sort       = (t3.tv_sec - t2.tv_sec)*1000000L + (t3.tv_nsec - t2.tv_nsec)/1000L;
        long us_levels     = (t5.tv_sec - t3.tv_sec)*1000000L + (t5.tv_nsec - t3.tv_nsec)/1000L;
        long us_total      = (t5.tv_sec - t0.tv_sec)*1000000L + (t5.tv_nsec - t0.tv_nsec)/1000L;
        fprintf(stderr, "[morton-prof step%d] morton=%ldus sort=%ldus 21-levels=%ldus total=%ldus\n",
                profileStepCount, us_morton, us_sort, us_levels, us_total);
        profileStepCount++;
    }

    return NBODY_CUDA_SUCCESS;
}

/* summarizationClear: invalidate cell mass/start fields and quad
 * moments in the range below the current treeStatus->bottom watermark.
 * Mirrors nbody_kernels.cl:1258-1296. */
__global__ void nbCUDASummarizationClearKernel(double* __restrict__ d_mass,
                                               int*    __restrict__ d_start,
                                               double* __restrict__ d_quadXX,
                                               double* __restrict__ d_quadXY,
                                               double* __restrict__ d_quadXZ,
                                               double* __restrict__ d_quadYY,
                                               double* __restrict__ d_quadYZ,
                                               double* __restrict__ d_quadZZ,
                                               const struct NBodyCUDATreeStatus* d_treeStatus,
                                               int nNode)
{
    __shared__ int bottom;
    if (threadIdx.x == 0) bottom = d_treeStatus->bottom;
    __syncthreads();

    const int inc = blockDim.x * gridDim.x;
    int k = (bottom & (-NBODY_CUDA_WARPSIZE)) + (int) (blockIdx.x * blockDim.x + threadIdx.x);
    if (k < bottom) k += inc;

    const double nanD = nan("");
    while (k < nNode)
    {
        d_mass[k]  = -1.0;
        d_start[k] = NBODY_CUDA_NULL_BODY;
        if (d_quadXX)               /* useQuad */
        {
            d_quadXX[k] = nanD;
            d_quadXY[k] = nanD;
            d_quadXZ[k] = nanD;
            d_quadYY[k] = nanD;
            d_quadYZ[k] = nanD;
            d_quadZZ[k] = nanD;
        }
        k += inc;
    }
}

extern "C" NBodyStatus_int nbCUDALaunchSummarizationClear(struct NBodyCUDABuffers* buffers,
                                                          int nbody,
                                                          int nNode)
{
    (void) nbody;
    if (!buffers || !buffers->d_treeStatus) return NBODY_CUDA_ERROR;
    const int block = NBODY_CUDA_BLOCK;
    const int grid  = buffers->numSMs > 0 ? buffers->numSMs : 1;
    nbCUDASummarizationClearKernel<<<grid, block>>>(buffers->d_masses,
                                                    buffers->d_start,
                                                    buffers->d_quadXX,
                                                    buffers->d_quadXY,
                                                    buffers->d_quadXZ,
                                                    buffers->d_quadYY,
                                                    buffers->d_quadYZ,
                                                    buffers->d_quadZZ,
                                                    buffers->d_treeStatus,
                                                    nNode);
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        fprintf(stderr, "[nbody_cuda] summarizationClear launch: %s\n", cudaGetErrorString(e));
        return NBODY_CUDA_ERROR;
    }
    return NBODY_CUDA_SUCCESS;
}

/* sort: walks the tree top-down assigning each leaf body a position in
 * the sort permutation. Bodies in the same leaf land contiguously,
 * which improves coalescing in the subsequent force kernel.
 * Mirrors the NVIDIA branch of nbody_kernels.cl:1541-1579. */
__global__ void nbCUDASortKernel(const int* __restrict__ d_count,
                                 int* __restrict__ d_start,
                                 int* __restrict__ d_sort,
                                 const int* __restrict__ d_child,
                                 const struct NBodyCUDATreeStatus* d_treeStatus,
                                 int nbody,
                                 int nNode)
{
    __shared__ int bottoms;
    if (threadIdx.x == 0) bottoms = d_treeStatus->bottom;
    __syncthreads();

    const int bottom = bottoms;
    const int dec = blockDim.x * gridDim.x;
    int k = nNode + 1 - dec + (int) (blockIdx.x * blockDim.x + threadIdx.x);

    while (k >= bottom)
    {
        int start = d_start[k];
        if (start >= 0)
        {
            #pragma unroll
            for (int i = 0; i < NBODY_CUDA_NSUB; ++i)
            {
                int ch = d_child[NBODY_CUDA_NSUB * k + i];
                if (ch >= nbody)            /* child is a cell */
                {
                    d_start[ch] = start;
                    start += d_count[ch];
                }
                else if (ch >= 0)           /* child is a body */
                {
                    d_sort[start] = ch;
                    ++start;
                }
            }
        }
        k -= dec;
    }
}

/* Identity permutation: d_sort[i] = i. Replaces the Barnes-Hut sort
 * for now; correctness-only, no spatial locality. */
__global__ void nbCUDASortIdentityKernel(int* __restrict__ d_sort, int nbody)
{
    const int stride = blockDim.x * gridDim.x;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < nbody;
         i += stride)
    {
        d_sort[i] = i;
    }
}

/* Serial top-down DFS sort: one thread, iterative stack-based walk
 * from root down. Produces a permutation of [0,nbody) where bodies
 * appear in tree-traversal order (siblings adjacent → spatial
 * locality for the force-walk). Slow vs a real parallel sort but
 * fast enough (~1 ms/step for ~80K cells) and provably correct. */
__global__ void nbCUDASortSerialDFSKernel(const int* __restrict__ d_child,
                                          int* __restrict__ d_sort,
                                          int nbody,
                                          int nNode)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) return;

    /* Stack of (cellIdx, nextChild). Sized generously: tree depth in
     * principle bounded by NBODY_CUDA_MAXDEPTH but in pathological
     * dwarf-cluster geometries it can briefly exceed; size to 64. */
    enum { STACK_SZ = 64 };
    int stackCell[STACK_SZ];
    int stackPos [STACK_SZ];

    int top = 0;
    stackCell[0] = nNode;     /* root */
    stackPos [0] = 0;

    int outIdx = 0;
    int overflowCount = 0;

    while (top >= 0)
    {
        int cell = stackCell[top];
        int pos  = stackPos [top];
        if (pos >= NBODY_CUDA_NSUB)
        {
            --top;
            continue;
        }
        stackPos[top] = pos + 1;
        int ch = d_child[NBODY_CUDA_NSUB * cell + pos];
        if (ch < 0) continue;          /* empty slot */
        if (ch < nbody)
        {
            if (outIdx < nbody) d_sort[outIdx++] = ch;
        }
        else if (top + 1 < STACK_SZ)
        {
            ++top;
            stackCell[top] = ch;
            stackPos [top] = 0;
        }
        else
        {
            ++overflowCount;           /* tree deeper than STACK_SZ */
        }
    }

    /* Diagnostic: print if anything's wrong. */
    if (overflowCount > 0 || outIdx != nbody)
    {
        printf("[sortDFS] outIdx=%d nbody=%d overflowCells=%d\n",
               outIdx, nbody, overflowCount);
    }

    /* Backstop: pad any remaining slots with body indices we haven't
     * placed yet. Should be a no-op if the tree truly contains every
     * body, but defends against tree-construction bugs that drop
     * leaves — without this the force kernel would walk garbage. */
    if (outIdx < nbody)
    {
        for (int b = 0; b < nbody && outIdx < nbody; ++b)
        {
            int found = 0;
            for (int j = 0; j < outIdx; ++j) {
                if (d_sort[j] == b) { found = 1; break; }
            }
            if (!found) d_sort[outIdx++] = b;
        }
    }
}

extern "C" NBodyStatus_int nbCUDALaunchSort(struct NBodyCUDABuffers* buffers,
                                            int nbody,
                                            int nNode)
{
    if (!buffers || !buffers->d_sort) return NBODY_CUDA_ERROR;
    const int block = NBODY_CUDA_BLOCK;
    const int grid  = buffers->numSMs > 0 ? buffers->numSMs : 1;

    /* Identity permutation. Spatial sort tested but produced large
     * per-metric divergence (different warp body groupings → different
     * accept-vote outcomes → different per-body forces). Identity is
     * the deterministic baseline. */
    nbCUDASortIdentityKernel<<<grid, block>>>(buffers->d_sort, nbody);
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        fprintf(stderr, "[nbody_cuda] sortIdentity launch: %s\n", cudaGetErrorString(e));
        return NBODY_CUDA_ERROR;
    }
    return NBODY_CUDA_SUCCESS;
}

/* ----- buildTree: octree insertion via atomic-CAS lock pattern -----
 *
 * One thread per body. Each thread descends the octree from the root,
 * choosing the child octant at every level by comparing the body's
 * position against the cell centre. When it finds a NULL or a
 * single-body slot, it tries to lock the slot via atomicCAS(slot, ch,
 * LOCK). The thread that wins the CAS owns that slot:
 *   - If the slot was NULL (-1), the body is inserted directly.
 *   - If the slot already held another body, new cells are
 *     allocated (atomicSub on _treeStatus->bottom) and both bodies
 *     are pushed down into separate octants until they no longer
 *     collide.
 * The final write to the slot (the new body index, or a "patch"
 * cell index) releases the lock.
 *
 * Mirrors nbody_kernels.cl:899-1159. The HAVE_CONSISTENT_MEMORY=TRUE
 * branch is taken throughout (see translation table). */
#define NBODY_CUDA_LOCK (-2)

__global__ void nbCUDABuildTreeKernel(double* __restrict__ d_posX,
                                      double* __restrict__ d_posY,
                                      double* __restrict__ d_posZ,
                                      double* __restrict__ d_critRadii,
                                      int*    __restrict__ d_child,
                                      struct NBodyCUDATreeStatus* d_treeStatus,
                                      int nbody,
                                      int nNode)
{
    /* Cached tree-root values, broadcast to every thread in the block. */
    __shared__ double radius;
    __shared__ double rootX, rootY, rootZ;
    __shared__ volatile int deadCount; /* threads in this block past end of work */

    int localMaxDepth = 1;
    bool newParticle = true;

    const unsigned int inc = blockDim.x * gridDim.x;
    int i = (int) (blockIdx.x * blockDim.x + threadIdx.x);

    if (threadIdx.x == 0)
    {
        radius = d_treeStatus->radius;
        rootX = d_posX[nNode];
        rootY = d_posY[nNode];
        rootZ = d_posZ[nNode];
        deadCount = 0;
    }
    __threadfence();
    __syncthreads();

    /* Each thread votes "dead" if it has no body to insert. The block
     * loops as long as any thread is still alive. */
    if (i >= nbody)
    {
        atomicAdd((int*) &deadCount, 1);
    }
    __syncthreads();

    /* These persist across iterations of the outer loop when the
     * thread is mid-insertion (i.e. retrying because of a CAS loss
     * or a not-yet-visible parent position). */
    double r = 0.0;
    double px = 0.0, py = 0.0, pz = 0.0;
    int j = 0, n = 0, depth = 1;
    bool posNotReady = false;

    while (deadCount != (int) blockDim.x)
    {
        if (i < nbody)
        {
            if (newParticle)
            {
                /* New body, so start traversing at root. */
                newParticle = false;
                posNotReady = false;

                px = d_posX[i];
                py = d_posY[i];
                pz = d_posZ[i];
                n = nNode;
                depth = 1;
                r = radius;

                /* Pick child octant of the root.
                 * Encoding matches CPU's nbSubIndex: bit 0 = Z, bit 1 = Y, bit 2 = X.
                 * Mismatched encoding produces a different per-cell child
                 * iteration order during summarization/force-walk, which
                 * gives 1-ULP-different CoM and per-body force from CPU
                 * due to FP non-associativity in the 8-child sum. */
                j = 0;
                if (rootZ <= pz) j |= 1;
                if (rootY <= py) j |= 2;
                if (rootX <= px) j |= 4;
            }

            int ch = *(volatile int*) &d_child[NBODY_CUDA_NSUB * n + j];
            /* Descend until we hit a body slot, an empty slot, or
             * a not-yet-visible cell position. Reads on cell pos and
             * child slots are volatile to bypass per-SM L1 — without
             * this, a thread can see the previous step's CoM (written
             * by summarization) rather than the freshly-set geometric
             * centre, and end up descending into a malformed branch. */
            while (ch >= nbody && !posNotReady && depth <= NBODY_CUDA_MAXDEPTH)
            {
                n = ch;
                ++depth;
                r *= 0.5;

                double pnx = *(volatile double*) &d_posX[n];
                double pny = *(volatile double*) &d_posY[n];
                double pnz = *(volatile double*) &d_posZ[n];

                /* NaN means the parent cell hasn't been finalized yet
                 * by the thread that created it; back off and retry. */
                posNotReady = isnan(pnx) || isnan(pny) || isnan(pnz);

                /* Octant encoding matches CPU's nbSubIndex (Z=bit0, Y=bit1, X=bit2). */
                j = 0;
                if (pnz <= pz) j |= 1;
                if (pny <= py) j |= 2;
                if (pnx <= px) j |= 4;
                ch = *(volatile int*) &d_child[NBODY_CUDA_NSUB * n + j];
            }

            /* Skip if the child slot is already locked by another
             * thread, or somehow points back at our own body. */
            if ((ch != NBODY_CUDA_LOCK) && (ch != i) && !posNotReady)
            {
                int locked = NBODY_CUDA_NSUB * n + j;

                /* Try to acquire the lock. atomicCAS returns the
                 * previous value; equality with `ch` means we won. */
                if (ch == atomicCAS(&d_child[locked], ch, NBODY_CUDA_LOCK))
                {
                    if (ch == -1)
                    {
                        /* Empty slot — insert the body directly. */
                        d_child[locked] = i;
                    }
                    else
                    {
                        /* Slot already held another body. Allocate
                         * fresh cells and push both bodies down
                         * until they fall in different octants. */
                        int patch = -1;
                        do
                        {
                            ++depth;

                            int cell = atomicSub(&d_treeStatus->bottom, 1) - 1;
                            if (cell <= nbody)
                            {
                                /* Out of cell node slots. */
                                d_treeStatus->errorCode = 1; /* NBODY_KERNEL_CELL_OVERFLOW */
                                d_treeStatus->bottom = nNode;
                            }
                            patch = max(patch, cell);

                            /* Mark this cell "not yet written" with NaN
                             * positions before any other thread can find
                             * it via d_child. Cells get recycled across
                             * steps (atomicSub returns same indices), so
                             * d_posX[cell] still holds last step's CoM
                             * unless we explicitly clear it. Concurrent
                             * readers see NaN and back off via the
                             * posNotReady path. Guard: only valid cells
                             * (> nbody); overflow puts cell in the body
                             * region and writing NaN there would corrupt
                             * a real body's position. */
                            if (cell > nbody)
                            {
                                d_posX[cell] = nan("");
                                d_posY[cell] = nan("");
                                d_posZ[cell] = nan("");
                                __threadfence();
                            }

                            /* New cell C is a child of n at depth D+1.
                             * C's size = r/2 (where r is n's size). C's
                             * center = n.center + sign * (n_size/4).
                             *
                             * CPU equivalent (calcOffset / nbInitMidpoint):
                             *   newPos = qPos + 0.25 * qsize * sign
                             *   newCellSize = 0.5 * qsize
                             *
                             * Earlier CUDA code stored r BEFORE halving as
                             * d_critRadii[cell] (= parent's size, WRONG —
                             * summarization reads this as the cell's psize)
                             * AND used `r` (= cell_size after halving) as
                             * the offset (giving 2x the correct offset =
                             * cell_size instead of cell_size/2). Both bugs
                             * fixed below: write child's true size into
                             * d_critRadii, offset by parent_size/4. */
                            double nx = d_posX[n];
                            double ny = d_posY[n];
                            double nz = d_posZ[n];

                            const double offset = 0.25 * r;  /* parent_size / 4 */
                            const double cellSize = 0.5 * r; /* child's actual size */

                            /* TODO: parameterize criterion */
                            d_critRadii[cell] = cellSize;

                            double x = nx + (px < nx ? -offset : offset);
                            double y = ny + (py < ny ? -offset : offset);
                            double z = nz + (pz < nz ? -offset : offset);

                            r = cellSize;  /* for next loop iteration */

                            d_posX[cell] = x;
                            d_posY[cell] = y;
                            d_posZ[cell] = z;

                            /* Splice intermediate cells into the
                             * tree as we go (only the topmost cell
                             * will become the value placed back into
                             * the locked slot at end). */
                            if (patch != cell)
                            {
                                d_child[NBODY_CUDA_NSUB * n + j] = cell;
                            }

                            double pchx = d_posX[ch];
                            double pchy = d_posY[ch];
                            double pchz = d_posZ[ch];

                            /* Octant encoding matches CPU's nbSubIndex (Z=bit0, Y=bit1, X=bit2). */
                            j = 0;
                            if (z <= pchz) j |= 1;
                            if (y <= pchy) j |= 2;
                            if (x <= pchx) j |= 4;

                            d_child[NBODY_CUDA_NSUB * cell + j] = ch;

                            __threadfence();

                            n = cell;
                            /* Octant encoding matches CPU's nbSubIndex (Z=bit0, Y=bit1, X=bit2). */
                            j = 0;
                            if (z <= pz) j |= 1;
                            if (y <= py) j |= 2;
                            if (x <= px) j |= 4;

                            ch = d_child[NBODY_CUDA_NSUB * n + j];
                            /* Loop until the two bodies separate or
                             * we run out of depth. */
                        }
                        while (ch >= 0 && depth <= NBODY_CUDA_MAXDEPTH);

                        d_child[NBODY_CUDA_NSUB * n + j] = i;
                        __threadfence();
                        /* Releasing the lock by writing the topmost
                         * patch-cell index into the original slot. */
                        d_child[locked] = patch;
                    }
                    __threadfence();

                    localMaxDepth = max(depth, localMaxDepth);
                    /* HAVE_CONSISTENT_MEMORY branch: advance to next
                     * body in this thread's slice. */
                    i += (int) inc;
                    newParticle = true;
                    if (i >= nbody)
                    {
                        atomicAdd((int*) &deadCount, 1);
                    }
                }
            }
        }

        /* Throttle so other warps can finish their loads before the
         * next round of contended atomicCAS attempts. */
        __threadfence();
        __syncthreads();
    }

    atomicMax(&d_treeStatus->maxDepth, localMaxDepth);
}

extern "C" NBodyStatus_int nbCUDALaunchBuildTree(struct NBodyCUDABuffers* buffers,
                                                 int nbody,
                                                 int nNode)
{
    if (!buffers || !buffers->d_treeStatus || !buffers->d_child) return NBODY_CUDA_ERROR;
    if (nbody != buffers->nbody) return NBODY_CUDA_ERROR;

    /* Dispatch: NBODY_BUILDTREE_MORTON env flag (read once, cached)
     * selects the deterministic Morton-based parallel tree builder
     * instead of the legacy atomicCAS path. Default = legacy. The
     * Morton path is deterministic across runs regardless of CUDA
     * runtime scheduling, which is required for opt #8 elaborate
     * (pinned + async D->H of bodies for bestLikelihood eval). */
    {
        static int useMortonCached = -1;
        if (useMortonCached < 0) {
            const char* env = getenv("NBODY_BUILDTREE_MORTON");
            useMortonCached = (env && env[0] && env[0] != '0') ? 1 : 0;
        }
        if (useMortonCached) {
            return nbCUDABuildTreeMorton(buffers, nbody, nNode);
        }
    }

    const int block = NBODY_CUDA_BLOCK;
    /* OpenCL ws->blocks[2] = 2 * maxCompUnits; mirror that here. */
    int grid = 2 * (buffers->numSMs > 0 ? buffers->numSMs : 1);
    if (grid < 1) grid = 1;

    nbCUDABuildTreeKernel<<<grid, block>>>(buffers->d_posX,
                                           buffers->d_posY,
                                           buffers->d_posZ,
                                           buffers->d_critRadii,
                                           buffers->d_child,
                                           buffers->d_treeStatus,
                                           nbody,
                                           nNode);
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        fprintf(stderr, "[nbody_cuda] buildTree launch: %s\n", cudaGetErrorString(e));
        return NBODY_CUDA_ERROR;
    }
    return NBODY_CUDA_SUCCESS;
}

/* ----- summarization: bottom-up cell COM/mass aggregation -----
 *
 * Each thread takes a cell index k and tries to compute its centre
 * of mass and total mass from its children. A cell is "ready" when
 * all of its children have non-negative mass (bodies are always
 * ready; cells become ready as their own COMs are summarized below).
 * If any child mass is still -1.0 (the "not yet computed" sentinel
 * set by summarizationClear), the thread caches the missing-child
 * indices in shared memory and polls them until they all arrive.
 * Threads always start at the deepest cells (largest indices, near
 * NNODE) so leaves' parents can finish quickly.
 *
 * Mirrors nbody_kernels.cl:1299-1510. */
__global__ void nbCUDASummarizationKernel(double* __restrict__ d_posX,
                                          double* __restrict__ d_posY,
                                          double* __restrict__ d_posZ,
                                          double* __restrict__ d_masses,
                                          int*    __restrict__ d_count,
                                          int*    __restrict__ d_child,
                                          double* __restrict__ d_critRadii,
                                          double* __restrict__ d_quadXX,
                                          struct NBodyCUDATreeStatus* d_treeStatus,
                                          int nbody,
                                          int nNode,
                                          double theta)
{
    __shared__ int bottom;
    __shared__ volatile int child[NBODY_CUDA_NSUB * NBODY_CUDA_BLOCK];
    __shared__ double rootSize;

    if (threadIdx.x == 0)
    {
        rootSize = d_treeStatus->radius;
        bottom = d_treeStatus->bottom;
    }
    __threadfence();
    __syncthreads();

    const int inc = blockDim.x * gridDim.x;
    int k = (bottom & (-NBODY_CUDA_WARPSIZE)) + (int) (blockIdx.x * blockDim.x + threadIdx.x);
    if (k < bottom) k += inc;

    int missing = 0;
    /* Per-cell accumulators; persist across the missing-child poll loop. */
    double cm = 0.0, px = 0.0, py = 0.0, pz = 0.0;
    int    cnt = 0;
    /* Snapshot mass+pos of children in compacted slots [0..nchildren-1]
     * during the fresh and poll passes. The final aggregation iterates
     * 0..nchildren-1 in fixed order, so the result is independent of
     * which children were ready in the fresh vs the poll pass — that
     * order varies between runs (CUDA scheduling) and breaks
     * reproducibility. After this fix, two runs with the same input
     * give bit-identical CoMs / Rcrit2 / per-body forces / likelihoods. */
    int    cell_chld[NBODY_CUDA_NSUB];
    double cell_m[NBODY_CUDA_NSUB];
    double cell_x[NBODY_CUDA_NSUB];
    double cell_y[NBODY_CUDA_NSUB];
    double cell_z[NBODY_CUDA_NSUB];
    int    cell_n = 0;

    while (k <= nNode)
    {
        double m = 0.0;
        int    ch;

        /* HAVE_CONSISTENT_MEMORY branch: always enter; sentinel
         * already filtered by the bottom-up start k. */
        {
            if (missing == 0)
            {
                /* Fresh cell: zero accumulators and scan all 8 children. */
                cm = 0.0;
                px = 0.0;
                py = 0.0;
                pz = 0.0;
                cnt = 0;
                int j = 0;

                #pragma unroll
                for (int i = 0; i < NBODY_CUDA_NSUB; ++i)
                {
                    ch = d_child[NBODY_CUDA_NSUB * k + i];
                    if (ch >= 0)
                    {
                        if (i != j)
                        {
                            /* Compact non-empty children to the front;
                             * later passes (force walk) rely on this. */
                            d_child[NBODY_CUDA_NSUB * k + i] = -1;
                            d_child[NBODY_CUDA_NSUB * k + j] = ch;
                        }

                        cell_chld[j] = ch;
                        m = d_masses[ch];
                        if (m < 0.0)
                        {
                            /* Cache child index so we can poll it later. */
                            child[NBODY_CUDA_BLOCK * missing + threadIdx.x] = ch;
                            ++missing;
                            cell_m[j] = -1.0;  /* sentinel: fill in poll */
                        }
                        else
                        {
                            if (ch >= nbody) cnt += d_count[ch] - 1;
                            cell_m[j] = m;
                            cell_x[j] = d_posX[ch];
                            cell_y[j] = d_posY[ch];
                            cell_z[j] = d_posZ[ch];
                        }
                        ++j;
                    }
                }
                cell_n = j;
                __threadfence();
                __threadfence_block();
                cnt += j;
            }

            /* Poll missing children with volatile reads. Capture into
             * the same compacted slot. */
            if (missing != 0)
            {
                do
                {
                    ch = child[NBODY_CUDA_BLOCK * (missing - 1) + threadIdx.x];
                    m = *(volatile double*) &d_masses[ch];
                    if (m >= 0.0)
                    {
                        --missing;
                        if (ch >= nbody) cnt += d_count[ch] - 1;
                        for (int s = 0; s < cell_n; ++s)
                        {
                            if (cell_chld[s] == ch && cell_m[s] < 0.0)
                            {
                                cell_m[s] = m;
                                cell_x[s] = *(volatile double*) &d_posX[ch];
                                cell_y[s] = *(volatile double*) &d_posY[ch];
                                cell_z[s] = *(volatile double*) &d_posZ[ch];
                                break;
                            }
                        }
                    }
                }
                while ((m >= 0.0) && (missing != 0));
            }

            if (missing == 0)
            {
                /* Final aggregation in fixed compacted-octant order
                 * 0..cell_n-1. This ordering is identical across runs
                 * (because compaction preserves octant order) and
                 * matches CPU's hackCofM iteration of Subp[0..7]. */
                cm = 0.0;
                px = 0.0;
                py = 0.0;
                pz = 0.0;
                #pragma unroll
                for (int s = 0; s < NBODY_CUDA_NSUB; ++s)
                {
                    if (s >= cell_n) break;
                    cm += cell_m[s];
                    /* Separate mul + add (matches CPU mw_incaddv_s). */
                    px += cell_m[s] * cell_x[s];
                    py += cell_m[s] * cell_y[s];
                    pz += cell_m[s] * cell_z[s];
                }

                /* All children ready — finalize this cell. */
                d_count[k] = cnt;
                double cx = d_posX[k];  /* geometric centre saved by buildTree */
                double cy = d_posY[k];
                double cz = d_posZ[k];

                /* TODO: parameterize criterion (TREECODE assumed). */
                double psize = d_critRadii[k];

                /* Direct division to bit-match CPU's mw_incdivs:
                 *   cmpos.x /= Mass(p)   (single rounded fdiv per axis).
                 * Reciprocal-then-multiply (1/cm * px) drifts by 1 ULP
                 * vs CPU. Verified by CPU disassembly: three fdiv instrs. */
                px /= cm;
                py /= cm;
                pz /= cm;

                /* Opening criterion radius squared. TREECODE branch. */
                double rc2;
                if (theta == 0.0)
                {
                    double twoR = 2.0 * rootSize;
                    rc2 = twoR * twoR;
                }
                else
                {
                    /* TODO: parameterize criterion */
                    double dx = px - cx;
                    double dy = py - cy;
                    double dz = pz - cz;
                    /* Separate ops to match CPU's mw_distv(cmpos, Pos(p)). */
                    double dr = sqrt(dx*dx + dy*dy + dz*dz);
                    double rc = (psize / theta) + dr;
                    rc2 = rc * rc;
                }

                /* Tree-structure sanity check (TREECODE/SW93). */
                bool xTest = (px < cx - psize) || (px > cx + psize);
                bool yTest = (py < cy - psize) || (py > cy + psize);
                bool zTest = (pz < cz - psize) || (pz > cz + psize);
                if (xTest || yTest || zTest)
                {
                    d_treeStatus->errorCode = 2; /* NBODY_KERNEL_TREE_STRUCTURE_ERROR */
                }

                d_posX[k] = px;
                d_posY[k] = py;
                d_posZ[k] = pz;

                d_critRadii[k] = rc2;

                if (d_quadXX)
                {
                    /* Mark "not yet computed" for the quadMoments
                     * pass; that kernel keys off isnan(quadXX[k]). */
                    d_quadXX[k] = nan("");
                }

                __threadfence();  /* Make data visible before publishing the mass. */
                d_masses[k] = cm;

                k += inc;  /* Move on to next cell */
            }
        }
    }
}

extern "C" NBodyStatus_int nbCUDALaunchSummarization(struct NBodyCUDABuffers* buffers,
                                                     int nbody,
                                                     int nNode)
{
    if (!buffers || !buffers->d_treeStatus) return NBODY_CUDA_ERROR;
    if (nbody != buffers->nbody) return NBODY_CUDA_ERROR;

    const int block = NBODY_CUDA_BLOCK;
    int grid = 2 * (buffers->numSMs > 0 ? buffers->numSMs : 1);
    if (grid < 1) grid = 1;

    /* TODO: thread theta through from NBodyCtx */
    const double theta = 1.0;

    nbCUDASummarizationKernel<<<grid, block>>>(buffers->d_posX,
                                               buffers->d_posY,
                                               buffers->d_posZ,
                                               buffers->d_masses,
                                               buffers->d_count,
                                               buffers->d_child,
                                               buffers->d_critRadii,
                                               buffers->d_quadXX,
                                               buffers->d_treeStatus,
                                               nbody,
                                               nNode,
                                               theta);
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        fprintf(stderr, "[nbody_cuda] summarization launch: %s\n", cudaGetErrorString(e));
        return NBODY_CUDA_ERROR;
    }
    return NBODY_CUDA_SUCCESS;
}

/* ----- quadMoments: bottom-up quadrupole moment aggregation -----
 *
 * Same shape as summarization but for the symmetric quadrupole
 * tensor. A cell's quadXX is set to NaN by summarization to flag
 * "not yet computed"; this kernel sets it to a real value once all
 * descendants' moments are folded in. */

/* Add 6 components of symmetric quadrupole tensor in place. */
__device__ __forceinline__ void incAddQuadMatrix(double* xx, double* xy, double* xz,
                                                 double* yy, double* yz, double* zz,
                                                 double bxx, double bxy, double bxz,
                                                 double byy, double byz, double bzz)
{
    *xx += bxx;
    *xy += bxy;
    *xz += bxz;
    *yy += byy;
    *yz += byz;
    *zz += bzz;
}

/* Reduce-style quadrupole: contribution from a single child mass at
 * chCM relative to the parent cell's centre kp. Mirrors the OpenCL
 * `quadCalc` (nbody_kernels.cl:1595-1612). */
__device__ __forceinline__ void quadCalc(double* qxx, double* qxy, double* qxz,
                                         double* qyy, double* qyz, double* qzz,
                                         double chx, double chy, double chz, double chw,
                                         double kpx, double kpy, double kpz)
{
    double drx = chx - kpx;
    double dry = chy - kpy;
    double drz = chz - kpz;
    /* Separate ops (no fma) to bit-match CPU's mw_sqrv in hackQuad. */
    double drSq = drx*drx + dry*dry + drz*drz;

    *qxx = chw * (3.0 * (drx * drx) - drSq);
    *qxy = chw * (3.0 * (drx * dry));
    *qxz = chw * (3.0 * (drx * drz));
    *qyy = chw * (3.0 * (dry * dry) - drSq);
    *qyz = chw * (3.0 * (dry * drz));
    *qzz = chw * (3.0 * (drz * drz) - drSq);
}

__global__ void nbCUDAQuadMomentsKernel(double* __restrict__ d_posX,
                                        double* __restrict__ d_posY,
                                        double* __restrict__ d_posZ,
                                        double* __restrict__ d_masses,
                                        int*    __restrict__ d_child,
                                        double* __restrict__ d_quadXX,
                                        double* __restrict__ d_quadXY,
                                        double* __restrict__ d_quadXZ,
                                        double* __restrict__ d_quadYY,
                                        double* __restrict__ d_quadYZ,
                                        double* __restrict__ d_quadZZ,
                                        struct NBodyCUDATreeStatus* d_treeStatus,
                                        int nbody,
                                        int nNode)
{
    __shared__ int bottom;
    __shared__ volatile int child[NBODY_CUDA_NSUB * NBODY_CUDA_BLOCK];
    __shared__ int maxDepth;

    if (threadIdx.x == 0)
    {
        bottom = d_treeStatus->bottom;
        maxDepth = d_treeStatus->maxDepth;
    }
    __threadfence();
    __syncthreads();

    const int inc = blockDim.x * gridDim.x;
    int k = (bottom & (-NBODY_CUDA_WARPSIZE)) + (int) (blockIdx.x * blockDim.x + threadIdx.x);
    if (k < bottom) k += inc;

    if (maxDepth > NBODY_CUDA_MAXDEPTH + 1)
    {
        d_treeStatus->errorCode = maxDepth;
        return;
    }

    int missing = 0;
    /* Accumulators persist across the missing-child poll loop. */
    double kqxx = 0.0, kqxy = 0.0, kqxz = 0.0;
    double kqyy = 0.0, kqyz = 0.0;
    double kqzz = 0.0;
    double kpx = 0.0, kpy = 0.0, kpz = 0.0;
    /* Per-cell snapshot of children: index, mass, pos, and (for cells)
     * the child's already-computed quadXX..ZZ. After all children
     * ready, aggregate in fixed octant order so the result is
     * deterministic across runs. */
    int    qc_chld[NBODY_CUDA_NSUB];
    double qc_chx[NBODY_CUDA_NSUB], qc_chy[NBODY_CUDA_NSUB], qc_chz[NBODY_CUDA_NSUB];
    double qc_chw[NBODY_CUDA_NSUB];
    double qc_qxx[NBODY_CUDA_NSUB], qc_qxy[NBODY_CUDA_NSUB], qc_qxz[NBODY_CUDA_NSUB];
    double qc_qyy[NBODY_CUDA_NSUB], qc_qyz[NBODY_CUDA_NSUB], qc_qzz[NBODY_CUDA_NSUB];
    int    qc_n = 0;

    while (k <= nNode)
    {
        int ch;
        double qChxx = 0.0;  /* sentinel value used in poll-loop while-condition */

        kpx = d_posX[k];
        kpy = d_posY[k];
        kpz = d_posZ[k];

        /* HAVE_CONSISTENT_MEMORY branch: always enter (cells with
         * already-computed quad moments were filtered by the
         * bottom-up start index). */
        {
            if (missing == 0)
            {
                int j = 0;
                #pragma unroll
                for (int i = 0; i < NBODY_CUDA_NSUB; ++i)
                {
                    ch = d_child[NBODY_CUDA_NSUB * k + i];
                    if (ch >= 0)
                    {
                        qc_chld[j] = ch;
                        if (ch < nbody)  /* isBody */
                        {
                            qc_chx[j] = d_posX[ch];
                            qc_chy[j] = d_posY[ch];
                            qc_chz[j] = d_posZ[ch];
                            qc_chw[j] = d_masses[ch];
                            /* NaN for child quad → "no quad recurrence". */
                            qc_qxx[j] = nan("");
                        }
                        else /* isCell */
                        {
                            qChxx = d_quadXX[ch];
                            if (!isnan(qChxx))
                            {
                                /* Already ready — snapshot now. */
                                qc_chx[j] = d_posX[ch];
                                qc_chy[j] = d_posY[ch];
                                qc_chz[j] = d_posZ[ch];
                                qc_chw[j] = d_masses[ch];
                                qc_qxx[j] = qChxx;
                                qc_qxy[j] = d_quadXY[ch];
                                qc_qxz[j] = d_quadXZ[ch];
                                qc_qyy[j] = d_quadYY[ch];
                                qc_qyz[j] = d_quadYZ[ch];
                                qc_qzz[j] = d_quadZZ[ch];
                            }
                            else
                            {
                                /* Not ready — push to missing and
                                 * mark slot to fill in poll. */
                                child[NBODY_CUDA_BLOCK * missing + threadIdx.x] = ch;
                                ++missing;
                                qc_qxx[j] = nan("missing");
                            }
                        }
                        ++j;
                    }
                }
                qc_n = j;
                __threadfence();
                __threadfence_block();
            }

            /* Poll missing children with volatile reads. Snapshot
             * into the same compacted slot. */
            if (missing != 0)
            {
                do
                {
                    ch = child[NBODY_CUDA_BLOCK * (missing - 1) + threadIdx.x];
                    qChxx = *(volatile double*) &d_quadXX[ch];
                    if (!isnan(qChxx))
                    {
                        --missing;
                        for (int s = 0; s < qc_n; ++s)
                        {
                            if (qc_chld[s] == ch && isnan(qc_qxx[s]))
                            {
                                qc_chx[s] = *(volatile double*) &d_posX[ch];
                                qc_chy[s] = *(volatile double*) &d_posY[ch];
                                qc_chz[s] = *(volatile double*) &d_posZ[ch];
                                qc_chw[s] = *(volatile double*) &d_masses[ch];
                                qc_qxx[s] = qChxx;
                                qc_qxy[s] = *(volatile double*) &d_quadXY[ch];
                                qc_qxz[s] = *(volatile double*) &d_quadXZ[ch];
                                qc_qyy[s] = *(volatile double*) &d_quadYY[ch];
                                qc_qyz[s] = *(volatile double*) &d_quadYZ[ch];
                                qc_qzz[s] = *(volatile double*) &d_quadZZ[ch];
                                break;
                            }
                        }
                    }
                }
                while ((!isnan(qChxx)) && (missing != 0));
            }

            /* Final aggregation in fixed octant 0..qc_n-1 order
             * (deterministic across runs). */
            if (missing == 0)
            {
                kqxx = kqxy = kqxz = 0.0;
                kqyy = kqyz = 0.0;
                kqzz = 0.0;
                for (int s = 0; s < qc_n; ++s)
                {
                    double quadxx = 0.0, quadxy = 0.0, quadxz = 0.0;
                    double quadyy = 0.0, quadyz = 0.0;
                    double quadzz = 0.0;
                    quadCalc(&quadxx, &quadxy, &quadxz,
                             &quadyy, &quadyz, &quadzz,
                             qc_chx[s], qc_chy[s], qc_chz[s], qc_chw[s],
                             kpx, kpy, kpz);
                    /* If child is a cell, its quadXX..ZZ are non-NaN;
                     * add the recurrence term. Bodies have NaN sentinel. */
                    if (qc_chld[s] >= nbody)
                    {
                        incAddQuadMatrix(&quadxx, &quadxy, &quadxz,
                                         &quadyy, &quadyz, &quadzz,
                                         qc_qxx[s], qc_qxy[s], qc_qxz[s],
                                         qc_qyy[s], qc_qyz[s], qc_qzz[s]);
                    }
                    incAddQuadMatrix(&kqxx, &kqxy, &kqxz,
                                     &kqyy, &kqyz, &kqzz,
                                     quadxx, quadxy, quadxz,
                                     quadyy, quadyz, quadzz);
                }
            }

            if (missing == 0)
            {
                /* Publish the off-diagonal/non-XX entries first; the
                 * XX write below acts as the "ready" flag for any
                 * other cell waiting on this one. */
                d_quadXY[k] = kqxy;
                d_quadXZ[k] = kqxz;
                d_quadYY[k] = kqyy;
                d_quadYZ[k] = kqyz;
                d_quadZZ[k] = kqzz;

                __threadfence();
                d_quadXX[k] = kqxx;
                __threadfence();

                k += inc;
            }
        }
    }
}

extern "C" NBodyStatus_int nbCUDALaunchQuadMoments(struct NBodyCUDABuffers* buffers,
                                                   int nNode)
{
    if (!buffers || !buffers->d_treeStatus || !buffers->d_quadXX) return NBODY_CUDA_ERROR;

    const int block = NBODY_CUDA_BLOCK;
    int grid = 2 * (buffers->numSMs > 0 ? buffers->numSMs : 1);
    if (grid < 1) grid = 1;

    nbCUDAQuadMomentsKernel<<<grid, block>>>(buffers->d_posX,
                                             buffers->d_posY,
                                             buffers->d_posZ,
                                             buffers->d_masses,
                                             buffers->d_child,
                                             buffers->d_quadXX,
                                             buffers->d_quadXY,
                                             buffers->d_quadXZ,
                                             buffers->d_quadYY,
                                             buffers->d_quadYZ,
                                             buffers->d_quadZZ,
                                             buffers->d_treeStatus,
                                             buffers->nbody,
                                             nNode);
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        fprintf(stderr, "[nbody_cuda] quadMoments launch: %s\n", cudaGetErrorString(e));
        return NBODY_CUDA_ERROR;
    }
    return NBODY_CUDA_SUCCESS;
}

/* Quad packing: copy the 6 d_quad** arrays (cell-major layout) into
 * a single d_quadPacked array (cell N's 6 components in slots
 * N*8 + 0..5, slots 6..7 padded). 64-byte stride per cell so the
 * full quad fits in one cache line. ForceTree reads from
 * d_quadPacked instead of 6 separate arrays — saves 5 cache-line
 * loads per accepted-cell visit on long-WU dense regions.
 *
 * One thread per cell, only cells in [bottom, nNode] need real data
 * (others are unused tree slots). Pad slots 6..7 to keep the
 * cache-line layout regular. */
__global__ void nbCUDAQuadPackKernel(
    const double* __restrict__ d_quadXX,
    const double* __restrict__ d_quadXY,
    const double* __restrict__ d_quadXZ,
    const double* __restrict__ d_quadYY,
    const double* __restrict__ d_quadYZ,
    const double* __restrict__ d_quadZZ,
    double* __restrict__ d_quadPacked,
    int nNode)
{
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k > nNode) return;
    const int o = k * 8;
    d_quadPacked[o + 0] = d_quadXX[k];
    d_quadPacked[o + 1] = d_quadXY[k];
    d_quadPacked[o + 2] = d_quadXZ[k];
    d_quadPacked[o + 3] = d_quadYY[k];
    d_quadPacked[o + 4] = d_quadYZ[k];
    d_quadPacked[o + 5] = d_quadZZ[k];
    d_quadPacked[o + 6] = 0.0;
    d_quadPacked[o + 7] = 0.0;
}

extern "C" NBodyStatus_int nbCUDALaunchQuadPack(struct NBodyCUDABuffers* buffers,
                                                int nNode)
{
    if (!buffers || !buffers->d_quadPacked || !buffers->d_quadXX) return NBODY_CUDA_ERROR;
    const int block = NBODY_CUDA_BLOCK;
    const int grid  = (nNode + 1 + block - 1) / block;
    nbCUDAQuadPackKernel<<<grid, block>>>(buffers->d_quadXX,
                                          buffers->d_quadXY,
                                          buffers->d_quadXZ,
                                          buffers->d_quadYY,
                                          buffers->d_quadYZ,
                                          buffers->d_quadZZ,
                                          buffers->d_quadPacked,
                                          nNode);
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        fprintf(stderr, "[nbody_cuda] quadPack launch: %s\n", cudaGetErrorString(e));
        return NBODY_CUDA_ERROR;
    }
    return NBODY_CUDA_SUCCESS;
}

/* Magic-number wire encoding for the leapfrog branch selector.
 * Both the force-tree kernel and the integration kernel compare the
 * runtime `branch` value against these. Defined here (before Phase 5)
 * so the force-tree kernel can reference them; the Phase 2 section
 * below uses the same constants. */
__device__ __constant__ double kFullKickBranch  = -125.0;
__device__ __constant__ double kSecondHalfBranch = -1024.0;

/* ----- Phase 5: Barnes-Hut tree-walk force kernel -----
 *
 * Translation of nbody_kernels.cl:1902-2253 (forceCalculation).
 *
 * Each warp acts as a single tree-walker: lane 0 of the warp drives a
 * shared per-warp depth-first stack, broadcasts the next child to every
 * lane via shared memory, and the whole warp evaluates the opening
 * criterion in lockstep. The CUDA `__all_sync(0xFFFFFFFFu, pred)`
 * primitive replaces the OpenCL `allBlock[]` shared-memory ballot — it
 * collapses the 32-lane vote into a single instruction with implicit
 * convergence.
 *
 * The persistent kernel stripes bodies across (block * grid) threads;
 * threads that finish their assigned bodies still participate in every
 * subsequent warp-vote so the warp stays in lockstep with whichever
 * lanes still have work. Each lane's accumulators (px, py, pz, ax, ay,
 * az, depth, skipSelf) become irrelevant once it's out of bodies, so it
 * doesn't matter what they vote — only that they vote.
 *
 * Hard-coded assumptions for this branch (matches the canonical
 * milkyway@home N-body workunit): TREECODE = TRUE, BH86 = FALSE,
 * SW93 = FALSE, EXACT = FALSE. The opening radius for every cell has
 * been pre-computed by the bounding-box / summarization kernels and
 * lives in d_critRadii[], so this kernel never references `theta`
 * directly. USE_QUAD is a runtime flag (`useQuad`); the shared quad
 * stack arrays are always allocated but only loaded/used when the flag
 * is set. USE_EXTERNAL_POTENTIAL is dropped — that lives in Phase 5b. */

__global__
void nbCUDAForceTreeKernel(
    const double* __restrict__ d_posX,
    const double* __restrict__ d_posY,
    const double* __restrict__ d_posZ,
    const double* __restrict__ d_masses,
    double* __restrict__ d_velX,
    double* __restrict__ d_velY,
    double* __restrict__ d_velZ,
    double* __restrict__ d_accX,
    double* __restrict__ d_accY,
    double* __restrict__ d_accZ,
    const int* __restrict__ d_sort,
    const int* __restrict__ d_child,
    const double* __restrict__ d_critRadii,
    const double* __restrict__ d_quadXX,
    const double* __restrict__ d_quadXY,
    const double* __restrict__ d_quadXZ,
    const double* __restrict__ d_quadYY,
    const double* __restrict__ d_quadYZ,
    const double* __restrict__ d_quadZZ,
    const double* __restrict__ d_quadPacked,  /* hot path uses this */
    struct NBodyCUDATreeStatus* d_treeStatus,
    int    nbody,
    int    nNode,
    double eps2,
    double timestep,
    int    useQuad,
    int    updateVel,
    double branch)
{
    /* Block layout: NBODY_CUDA_BLOCK / NBODY_CUDA_WARPSIZE warps,
     * each running an independent tree walk. Per-warp scratch is
     * indexed by `base = threadIdx.x / WARPSIZE`. */
    constexpr int kWarpsPerBlock = NBODY_CUDA_BLOCK / NBODY_CUDA_WARPSIZE;
    constexpr int kStackSlots    = NBODY_CUDA_MAXDEPTH * kWarpsPerBlock;

    /* Single per-block scalars set by thread 0. */
    __shared__ unsigned int maxDepth;
    __shared__ double rootCritRadius;

    /* (formerly per-warp child broadcast slots — now broadcast via
     * __shfl_sync register-to-register, which is faster than the
     * shared-mem write+read+__syncwarp triple.) */

    /* Per-warp DFS stacks (MAXDEPTH entries each). */
    __shared__ volatile int    pos [kStackSlots];
    __shared__ volatile int    node[kStackSlots];
    __shared__ volatile double dq  [kStackSlots];

    /* Quadrupole stacks: always allocated, only populated when
     * useQuad is non-zero. ~9.7 KB total — fits easily in shared mem. */
    __shared__ volatile double quadXX[kStackSlots];
    __shared__ volatile double quadXY[kStackSlots];
    __shared__ volatile double quadXZ[kStackSlots];
    __shared__ volatile double quadYY[kStackSlots];
    __shared__ volatile double quadYZ[kStackSlots];
    __shared__ volatile double quadZZ[kStackSlots];

    /* Root-cell quadrupole snapshot (one per block). */
    __shared__ double rootQXX, rootQXY, rootQXZ;
    __shared__ double rootQYY, rootQYZ;
    __shared__ double rootQZZ;

    /* Thread 0 reads the per-step tree status + root cell data once. */
    if (threadIdx.x == 0)
    {
        maxDepth = (unsigned int) d_treeStatus->maxDepth;

        if (useQuad)
        {
            /* All-volatile reads + collective NaN check, matching the
             * open-cell quad load below. */
            double qxx = *(volatile double*) &d_quadXX[nNode];
            double qxy = *(volatile double*) &d_quadXY[nNode];
            double qxz = *(volatile double*) &d_quadXZ[nNode];
            double qyy = *(volatile double*) &d_quadYY[nNode];
            double qyz = *(volatile double*) &d_quadYZ[nNode];
            double qzz = *(volatile double*) &d_quadZZ[nNode];
            if (isnan(qxx) || isnan(qxy) || isnan(qxz)
                || isnan(qyy) || isnan(qyz) || isnan(qzz))
            {
                rootQXX = rootQXY = rootQXZ = 0.0;
                rootQYY = rootQYZ = rootQZZ = 0.0;
            }
            else
            {
                rootQXX = qxx; rootQXY = qxy; rootQXZ = qxz;
                rootQYY = qyy; rootQYZ = qyz; rootQZZ = qzz;
            }
        }

        /* TREECODE branch: opening-criterion radius is per-cell, set
         * up by the summarization pass; we just read the root's value. */
        rootCritRadius = d_critRadii[nNode];

        /* Allow MAXDEPTH+1: buildTree can leave maxDepth one over due to
         * its increment-then-check loop. Bailing here would skip every
         * body's force calc and the simulation immediately blows up. */
        if (maxDepth > (unsigned int) (NBODY_CUDA_MAXDEPTH + 1))
        {
            d_treeStatus->errorCode = (int) maxDepth;
        }
    }
    __threadfence();
    __syncthreads();

    if (maxDepth > (unsigned int) (NBODY_CUDA_MAXDEPTH + 1))
    {
        return;
    }

    /* Per-warp scratch indices. */
    const unsigned int base  = threadIdx.x / NBODY_CUDA_WARPSIZE;
    const unsigned int sbase = base * NBODY_CUDA_WARPSIZE;
    const int          j     = (int) (base * NBODY_CUDA_MAXDEPTH);
    const bool         leader = (threadIdx.x == sbase);

    /* Persistent body stride: one pass per lane, then the lane jumps
     * forward by total launched threads. Inactive lanes still vote in
     * every __all_sync below to keep the warp's predicate mask sane. */
    const int stride = (int) (blockDim.x * gridDim.x);
    int k = (int) (blockIdx.x * blockDim.x + threadIdx.x);

    /* Cached per-body values (only meaningful when k < nbody). */
    double px = 0.0, py = 0.0, pz = 0.0;
    int    i  = -1;

    while (true)
    {
        /* Outer-loop convergence point: every lane in the warp
         * participates in the tree walk together, regardless of whether
         * it still has a body to process. */
        const bool alive = (k < nbody);
        if (alive)
        {
            i  = d_sort[k];      /* permuted body index */
            /* __ldg: read-only / texture-cache load. Bypasses the
             * -dlcm=cv L1-bypass set globally for buildTree
             * determinism; safe here because d_pos* is mutated by
             * the integration kernel which runs AFTER forceTree, so
             * the values are stable for the duration of this launch. */
            px = __ldg(&d_posX[i]);
            py = __ldg(&d_posY[i]);
            pz = __ldg(&d_posZ[i]);
        }
        else
        {
            i = -1;
        }

        /* Compute the warp-wide mask of lanes that still have work.
         * Used below in __all_sync so dead lanes don't pollute the
         * opening-criterion vote. Exit when no lane has work left. */
        const unsigned int liveMask = __ballot_sync(0xFFFFFFFFu, alive);
        if (liveMask == 0u)
        {
            break;
        }

        double ax = 0.0, ay = 0.0, az = 0.0;

        /* Per-lane skip depth — when this lane has accepted a cell at
         * iteration K (depth==K when the cell was processed), set
         * lane_skip_depth = K. The lane skips contributions while
         * depth > lane_skip_depth (warp descended into the cell's
         * subtree). When warp pops back so depth <= lane_skip_depth,
         * lane resets to active. */
        int lane_skip_depth = -1;

        /* Push root onto this warp's stack (lane 0 only). */
        int depth = j;
        if (leader)
        {
            node[j] = nNode;
            pos[j]  = 0;
            dq[j]   = rootCritRadius;

            if (useQuad)
            {
                quadXX[j] = rootQXX;
                quadXY[j] = rootQXY;
                quadXZ[j] = rootQXZ;
                quadYY[j] = rootQYY;
                quadYZ[j] = rootQYZ;
                quadZZ[j] = rootQZZ;
            }
        }
        /* Volta+ uses independent thread scheduling: lanes in a warp
         * may run out of step until an explicit sync. The OpenCL
         * mem_fence here was sufficient on AMD wavefronts (lockstep);
         * on CUDA we need __syncwarp so followers actually see the
         * leader's writes to node/pos/dq before reading them. */
        __syncwarp();

        bool skipSelf = false;
        do
        {
            int curPos;

            /* Stack frame at `depth` still has more children to process. */
            while ((curPos = pos[depth]) < NBODY_CUDA_NSUB)
            {
                int n = -1;
                double nx_loc = 0.0, ny_loc = 0.0, nz_loc = 0.0, nm_loc = 0.0;
                if (leader)
                {
                    /* Leader fetches the next child of node[depth] and
                     * its mass/position; broadcast to lanes via
                     * __shfl_sync below. */
                    n = d_child[NBODY_CUDA_NSUB * node[depth] + curPos];
                    pos[depth] = curPos + 1;
                    if (n >= 0)
                    {
                        /* Plain reads (no __ldg) — d_pos and d_masses
                         * for n >= nbody is the cell CoM written by
                         * summarization in THIS step; using __ldg
                         * here can read stale per-step values from
                         * the read-only cache (verified: produces
                         * delta ~0.02 on long WU). */
                        nx_loc = d_posX[n];
                        ny_loc = d_posY[n];
                        nz_loc = d_posZ[n];
                        nm_loc = d_masses[n];
                    }
                }
                /* Broadcast leader's values to all lanes via warp
                 * shuffle. __shfl_sync is a register-to-register move
                 * (single instruction) and provides warp-level
                 * synchronization, replacing the previous shared-mem
                 * + __syncwarp + shared-read triple. srcLane=0 means
                 * the warp's lane-0 (the leader). */
                n = __shfl_sync(0xFFFFFFFFu, n, 0);
                nx_loc = __shfl_sync(0xFFFFFFFFu, nx_loc, 0);
                ny_loc = __shfl_sync(0xFFFFFFFFu, ny_loc, 0);
                nz_loc = __shfl_sync(0xFFFFFFFFu, nz_loc, 0);
                nm_loc = __shfl_sync(0xFFFFFFFFu, nm_loc, 0);

                if (n >= 0)
                {
                    const double dx = nx_loc - px;
                    const double dy = ny_loc - py;
                    const double dz = nz_loc - pz;
                    double rSq = dx*dx + dy*dy + dz*dz;

                    /* Reset skip mode when warp has popped back at or
                     * above this lane's accept depth. */
                    if (lane_skip_depth >= 0 && depth <= lane_skip_depth)
                    {
                        lane_skip_depth = -1;
                    }

                    /* Per-lane open/accept against CHILD's own rc²
                     * (matches CPU's nbGravity per-body decision).
                     * Lanes in skip mode or with no body assigned sit
                     * out (don't contribute, don't vote). Warp pushes
                     * iff ANY live+active lane wants to open.
                     *
                     * IMPORTANT: use mask 0xFFFFFFFFu (all lanes), not
                     * liveMask. __any_sync's return value is UNDEFINED
                     * for lanes not in the mask — so dead lanes would
                     * see a different `warpOpens` than live lanes and
                     * the per-lane `++depth` would diverge, breaking
                     * the warp-uniform stack invariant. With full mask
                     * we suppress dead lanes' vote via `&& alive`. */
                    const bool isBody = (n < nbody);
                    const double childCrit = isBody ? 0.0 : d_critRadii[n];
                    const bool laneActive = alive && (lane_skip_depth < 0);
                    const bool laneAccepts = laneActive && (isBody || (rSq >= childCrit));
                    const bool laneWantsOpen = laneActive && !isBody && !(rSq >= childCrit);
                    const bool warpOpens = __any_sync(0xFFFFFFFFu, laneWantsOpen);
                    const bool accept = laneAccepts;

                    if (accept)
                    {
                        /* Single-particle (body or accepted cell)
                         * Plummer-softened gravity contribution.
                         * Separate ops (no fma) to bit-match CPU's
                         *   acc0.x += mor3 * dr.x;
                         * which compiles to mulsd+addsd under SSE2.
                         *
                         * v71: match CPU's TWO-divisions form exactly:
                         *   drab = sqrt(drSq); phii = Mass/drab; mor3 = phii/drSq
                         * Previous form `Mass / (rSq * r)` is 1 mul + 1 div,
                         * which differs from CPU's 2 div by ~1 ULP per cell
                         * visit. Compounds over thousands of cells per body
                         * per step over 64673 steps. */
                        rSq += eps2;
                        const double r    = sqrt(rSq);
                        const double phii = nm_loc / r;
                        const double ai   = phii / rSq;

                        ax += ai * dx;
                        ay += ai * dy;
                        az += ai * dz;

                        if (useQuad && n >= nbody)
                        {
                            /* Use child n's OWN quad (CPU's Quad(q) where
                             * q is the visited node), not the parent's
                             * (quadXX[depth] = node[depth] = parent of n
                             * in the OpenCL/CUDA convention). All-6 NaN
                             * check + zero fallback to defend against
                             * partial-init in any cell. */
                            /* Packed quad layout: cell n's 6 components
                             * at offset n*8 (slots 6..7 padded). One
                             * 64-byte cache-line load gets all 6 — vs
                             * 6 separate cache-line loads from the 6
                             * d_quad** arrays (each is a different
                             * memory region for cell n). */
                            const double* qp = d_quadPacked + (size_t)n * 8;
                            const double q_xx = qp[0];
                            const double q_xy = qp[1];
                            const double q_xz = qp[2];
                            const double q_yy = qp[3];
                            const double q_yz = qp[4];
                            const double q_zz = qp[5];

                            const bool qok = !(isnan(q_xx) || isnan(q_xy) || isnan(q_xz)
                                            || isnan(q_yy) || isnan(q_yz) || isnan(q_zz));
                            if (qok)
                            {
                                const double dr5inv = 1.0 / (rSq * rSq * r);

                                const double qdx = q_xx*dx + q_xy*dy + q_xz*dz;
                                const double qdy = q_xy*dx + q_yy*dy + q_yz*dz;
                                const double qdz = q_xz*dx + q_yz*dy + q_zz*dz;

                                const double drQdr = qdx*dx + qdy*dy + qdz*dz;

                                const double phiQuad = 2.5 * (dr5inv * drQdr) / rSq;

                                ax += phiQuad * dx;
                                ay += phiQuad * dy;
                                az += phiQuad * dz;

                                ax -= dr5inv * qdx;
                                ay -= dr5inv * qdy;
                                az -= dr5inv * qdz;
                            }
                        }

                        /* Self-interaction flag: dx,dy,dz are 0 so the
                         * accumulator was unchanged, but we record that
                         * it happened so the host can verify the body
                         * is still present in its own leaf cell. */
                        if (n == i)
                        {
                            skipSelf = true;
                        }

                        /* Mark this lane as done with this cell's subtree. */
                        if (!isBody)
                        {
                            lane_skip_depth = depth;
                        }
                    }

                    /* Warp-uniform descent: push iff any active lane
                     * voted to open. Lanes that already accepted sit
                     * out via lane_skip_depth. */
                    if (warpOpens)
                    {
                        ++depth;
                        if (leader)
                        {
                            node[depth] = n;
                            pos[depth]  = 0;
                            dq[depth]   = d_critRadii[n];

                            if (useQuad)
                            {
                                double qxx = *(volatile double*) &d_quadXX[n];
                                double qxy = *(volatile double*) &d_quadXY[n];
                                double qxz = *(volatile double*) &d_quadXZ[n];
                                double qyy = *(volatile double*) &d_quadYY[n];
                                double qyz = *(volatile double*) &d_quadYZ[n];
                                double qzz = *(volatile double*) &d_quadZZ[n];
                                if (isnan(qxx) || isnan(qxy) || isnan(qxz)
                                    || isnan(qyy) || isnan(qyz) || isnan(qzz))
                                {
                                    quadXX[depth] = 0.0;
                                    quadXY[depth] = 0.0;
                                    quadXZ[depth] = 0.0;
                                    quadYY[depth] = 0.0;
                                    quadYZ[depth] = 0.0;
                                    quadZZ[depth] = 0.0;
                                }
                                else
                                {
                                    quadXX[depth] = qxx;
                                    quadXY[depth] = qxy;
                                    quadXZ[depth] = qxz;
                                    quadYY[depth] = qyy;
                                    quadYZ[depth] = qyz;
                                    quadZZ[depth] = qzz;
                                }
                            }
                        }
                        __syncwarp();
                    }
                }
                else
                {
                    /* NULL slot: per OpenCL comment, all subsequent
                     * children at this level are also NULL — bail out
                     * one level (but never below this warp's stack base). */
                    depth = max(j, depth - 1);
                }
            }
            --depth;  /* pop */
        }
        while (depth >= j);

        /* Writeback / leapfrog correction — only meaningful when this
         * lane actually has a body. Inactive lanes did the tree walk
         * for warp-vote sake but skip the writes. */
        if (alive)
        {
            const double accX = d_accX[i];
            const double accY = d_accY[i];
            const double accZ = d_accZ[i];

            d_accX[i] = ax;
            d_accY[i] = ay;
            d_accZ[i] = az;

            /* FULL_KICK (interior re-kick) branch only: apply the
             * old-vs-new acceleration delta to v. The compare-by-double
             * matches the OpenCL kernel's magic-number wire encoding. */
            if (branch == kFullKickBranch)
            {
                double vx = d_velX[i];
                double vy = d_velY[i];
                double vz = d_velZ[i];

                /* Separate ops to bit-match CPU's bodyAdvanceVel:
                 *   dv = mw_mulvs(a, dtHalf);  mw_incaddv(Vel(p), dv);
                 * which is `vx += dtHalf * (ax - accX)` (mulsd + addsd). */
                const double dtHalf = 0.5 * timestep;
                vx += dtHalf * (ax - accX);
                vy += dtHalf * (ay - accY);
                vz += dtHalf * (az - accZ);

                if (updateVel)
                {
                    d_velX[i] = vx;
                    d_velY[i] = vy;
                    d_velZ[i] = vz;
                }
            }

            if (!skipSelf)
            {
                /* Tree incest: this body wasn't found in its own leaf,
                 * which means the tree is malformed at the workunit's
                 * tolerance. NBODY_KERNEL_TREE_INCEST = 4 (OR-flag). */
                d_treeStatus->errorCode = 4;
            }
        }

        /* Advance to the next assigned body. Per-lane advance is fine
         * because the warp-vote loop above is already convergent at
         * this point — every lane is past the tree walk. */
        k += stride;
    }
}

extern "C" NBodyStatus_int nbCUDALaunchForceTree(struct NBodyCUDABuffers* buffers,
                                                 int nbody,
                                                 int nNode,
                                                 double eps2,
                                                 double theta,
                                                 int useQuad,
                                                 int updateVel,
                                                 double timestep,
                                                 NBodyCUDAIntegrationPhase phase)
{
    /* `theta` is encoded into d_critRadii by the bounding-box / summary
     * pass, so the kernel itself never reads it. Accept it for API
     * symmetry with the OpenCL launcher. */
    (void) theta;

    if (!buffers || !buffers->d_treeStatus || !buffers->d_child ||
        !buffers->d_critRadii || !buffers->d_sort)
    {
        return NBODY_CUDA_ERROR;
    }
    if (nbody <= 0 || nbody != buffers->nbody) return NBODY_CUDA_ERROR;
    if (useQuad && !buffers->d_quadXX) return NBODY_CUDA_ERROR;

    const int block = NBODY_CUDA_BLOCK;
    int       grid  = 2 * (buffers->numSMs > 0 ? buffers->numSMs : 1);
    if (grid < 1) grid = 1;

    const double branch = (double) (int) phase;  /* preserves wire encoding */

    nbCUDAForceTreeKernel<<<grid, block>>>(buffers->d_posX,
                                           buffers->d_posY,
                                           buffers->d_posZ,
                                           buffers->d_masses,
                                           buffers->d_velX,
                                           buffers->d_velY,
                                           buffers->d_velZ,
                                           buffers->d_accX,
                                           buffers->d_accY,
                                           buffers->d_accZ,
                                           buffers->d_sort,
                                           buffers->d_child,
                                           buffers->d_critRadii,
                                           buffers->d_quadXX,
                                           buffers->d_quadXY,
                                           buffers->d_quadXZ,
                                           buffers->d_quadYY,
                                           buffers->d_quadYZ,
                                           buffers->d_quadZZ,
                                           buffers->d_quadPacked,
                                           buffers->d_treeStatus,
                                           nbody,
                                           nNode,
                                           eps2,
                                           timestep,
                                           useQuad,
                                           updateVel,
                                           branch);

    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess)
    {
        fprintf(stderr, "[nbody_cuda] forceTree launch: %s\n", cudaGetErrorString(e));
        return NBODY_CUDA_ERROR;
    }
    return NBODY_CUDA_SUCCESS;
}

/* ----- Phase 5b: external Milky Way + LMC potential kernel -----
 *
 * Adds the contribution of the Milky Way's static external potential
 * (bulge + 1 or 2 disk components + halo) and an optionally-active
 * moving LMC point-mass to the per-body acceleration. Runs after the
 * tree-walk force kernel and before integration. Mirrors the
 * inline-potential block at nbody_kernels.cl:2199-2225 plus the
 * potential model functions at lines 267-560 of the same file.
 *
 * The kernel itself is small; the work lives in the __device__
 * helpers below, one per supported potential model. Helpers all
 * return a 3-vector acceleration (ax,ay,az) by reference.
 */

#define NBODY_CUDA_SQR(x) ((x) * (x))

/* Spherical (bulge) models.
 * Hernquist:  acc = -mass / (r * (a + r)^2) * pos
 * Plummer:    acc = -mass / (a^2 + r^2)^(3/2) * pos    */
__device__ __forceinline__ void nbCUDAAccelHernquistSphere(
    double px, double py, double pz, double r,
    double mass, double scale,
    double* ax, double* ay, double* az)
{
    const double tmp = scale + r;
    const double c   = -mass / (r * (tmp * tmp));
    *ax += c * px;
    *ay += c * py;
    *az += c * pz;
}

__device__ __forceinline__ void nbCUDAAccelPlummerSphere(
    double px, double py, double pz, double r,
    double mass, double scale,
    double* ax, double* ay, double* az)
{
    const double tmp = sqrt(NBODY_CUDA_SQR(scale) + NBODY_CUDA_SQR(r));
    /* CPU computes mass / mw_pow(tmp, 3.0) where mw_pow is crlibm pow_rn.
     * crlibm's y=3 special case `x*(x*x)` only fires when x's low 32
     * mantissa bits are zero (the "yl == 0" check uses x's split, not
     * y's). For sqrt-derived tmp values the low bits are almost never
     * zero, so crlibm takes the general exp(3*log(x)) path producing a
     * correctly-rounded x^3. `tmp * tmp * tmp` does TWO multiplies and
     * is 1 ULP off the correctly-rounded result. Use vendored pow_rn
     * to bit-match CPU. */
    const double tmp3 = cuda_crlibm_pow_rn(tmp, 3.0);
    const double c   = -mass / tmp3;
    *ax += c * px;
    *ay += c * py;
    *az += c * pz;
}

/* Disk models.
 * Miyamoto-Nagai: closed-form 3D potential commonly used for the MW disk. */
__device__ __forceinline__ void nbCUDAAccelMiyamotoNagaiDisk(
    double px, double py, double pz,
    double mass, double a, double b,
    double* ax, double* ay, double* az)
{
    /* GCC with -O2 -fno-math-errno optimizes pow(x,2) -> x*x and
     * pow(x,0.5) -> sqrt(x). NVCC doesn't, so use x*x and sqrt
     * directly to actually match what CPU emits, not the textual
     * mw_pow calls. */
    const double zp  = sqrt(pz*pz + b*b);
    const double azp = a + zp;
    const double rp  = px*px + py*py + azp*azp;
    /* v72: combine vendored pow_rn with v71 BH gravity 2-div fix to
     * see if both fixes together close the gap (vs each in isolation
     * causing redistribution). */
    const double rth = cuda_crlibm_pow_rn(rp, 1.5);
    *ax += -mass * px / rth;
    *ay += -mass * py / rth;
    *az += -mass * pz * azp / (zp * rth);
}

/* Halo models.
 * Logarithmic: matches CPU's logHaloAccel formulation exactly:
 *   denom = d² + x² + y² + (z/q)²
 *   k     = -2 V² / denom
 *   ax    = k * x
 *   ay    = k * y
 *   az    = k * z / q²
 * Order matters for FP rounding — keep one denom for all three. */
__device__ __forceinline__ void nbCUDAAccelLogHalo(
    double px, double py, double pz,
    double vhalo, double scaleLength, double flattenZ,
    double* ax, double* ay, double* az)
{
    const double v0 = vhalo;
    const double q  = flattenZ;
    const double d  = scaleLength;
    /* CPU's mw_pow(x, 2.0) optimizes to x*x with -O2; use that. */
    const double zoq = pz / q;
    const double denom = d*d + px*px + py*py + zoq*zoq;
    const double k = -2.0 * v0 * v0 / denom;
    *ax += k * px;
    *ay += k * py;
    *az += k * pz / (q * q);
}

/* NFW halo (vhalo-parametrized). Mirrors CPU's nfwHaloAccel
 * (nbody_potential.c:675-684) operation order EXACTLY:
 *
 *   M = vhalo² * a / 0.2162165954
 *   ar = a + r
 *   c = (-M/r²) * (log(ar/a)/r - 1/ar)
 *
 * Earlier form `c = a*v²*(r - ar*log(...))/(0.2162... * r³ * ar)` is
 * algebraically equivalent but uses a different FP operation order.
 * That difference produces ~1 ULP per cell visit which compounds
 * chaotically in long-WU dwarf-disruption integrations, the same
 * pattern that affects NFWMass long-WU bin-flip divergence. */
__device__ __forceinline__ void nbCUDAAccelNFWHalo(
    double px, double py, double pz, double r,
    double vhalo, double scaleLength,
    double* ax, double* ay, double* az)
{
    const double a  = scaleLength;
    const double M  = NBODY_CUDA_SQR(vhalo) * a / 0.2162165954;
    const double ar = a + r;
    const double c  = (-M / (r * r)) * (cuda_crlibm_log_rn(ar / a) / r - 1.0 / ar);
    *ax += c * px;
    *ay += c * py;
    *az += c * pz;
}

/* NFWMass halo: NFW profile parametrized by enclosed mass instead of
 * vhalo. Mirrors CPU's NFWMHaloAccel (nbody_potential.c:622-632):
 *
 *   c = (-M / r²) * (log((a+r)/a)/r - 1/(a+r))
 *   acc = pos * c
 *
 * Operation order matches CPU's expression so per-step rounding
 * tracks crlibm log_rn → CUDA __device__ log to ~1 ULP.
 * Numerical note: the (log(ar/a)/r − 1/ar) cancellation loses
 * precision for very small r, but the upstream r-clamp in
 * nbExtAcceleration (r ≥ 2^−8) keeps us safely away from that
 * regime. The CPU has the same cancellation pattern. */
__device__ __forceinline__ void nbCUDAAccelNFWMassHalo(
    double px, double py, double pz, double r,
    double mass, double scaleLength,
    double* ax, double* ay, double* az)
{
    const double a  = scaleLength;
    const double M  = mass;
    const double ar = a + r;
    const double c  = (-M / (r * r)) * (cuda_crlibm_log_rn(ar / a) / r - 1.0 / ar);
    *ax += c * px;
    *ay += c * py;
    *az += c * pz;
}

/* Spherical NFW (Erkal variant). Mirrors CPU's
 * SphericalNFWerkalHaloAccel (nbody_potential.c:665-672) operation
 * order EXACTLY:
 *
 *   ar  = a + r
 *   tmp = M / (log(1 + 15.3) - 15.3/(1 + 15.3))   // c = 15.3 hard-coded
 *   c   = tmp * (r - ar*log(1 + r/a)) / (r³ * ar)
 *
 * The normalization constant `log(16.3) - 15.3/16.3` is computed
 * per call (rather than precomputed on host) so the FP rounding
 * matches CPU bit-for-bit — both call crlibm log_rn once, both
 * subtract the same way. */
__device__ __forceinline__ void nbCUDAAccelSphericalNFWerkalHalo(
    double px, double py, double pz, double r,
    double mass, double scaleLength,
    double* ax, double* ay, double* az)
{
    const double a   = scaleLength;
    const double M   = mass;
    const double ar  = a + r;
    const double tmp = M / (cuda_crlibm_log_rn(1.0 + 15.3) - (15.3 / (1.0 + 15.3)));
    const double c   = tmp * (r - (ar * cuda_crlibm_log_rn(1.0 + (r / a)))) / (r * r * r * ar);
    *ax += c * px;
    *ay += c * py;
    *az += c * pz;
}

/* Plummer LMC: same formula as Plummer bulge but with the LMC's
 * current position as the centre. Force on a body at `pos` from a
 * Plummer point-mass at `lmcPos`. */
__device__ __forceinline__ void nbCUDAAccelPlummerLMC(
    double px, double py, double pz,
    double lmcX, double lmcY, double lmcZ,
    double mass, double scale,
    double* ax, double* ay, double* az)
{
    const double dx   = lmcX - px;
    const double dy   = lmcY - py;
    const double dz   = lmcZ - pz;
    /* CPU plummerLmcAccel uses `mass / mw_pow(tmp, 3.0)`. The earlier
     * comment claimed crlibm's pow(x,3) takes a special-case
     * `x * (x * x)` path. Wrong: that path is gated on x's low-32
     * mantissa bits being zero (`yl = x_low; if (yl == 0)` in pow.c).
     * For sqrt-derived tmp the bits are almost never zero, so crlibm
     * uses the general exp(3*log(x)) path which is correctly-rounded
     * to a different value than `tmp * (tmp * tmp)` (1 ULP off, since
     * direct mul does two roundings). Use vendored pow_rn to match. */
    const double dist = sqrt(dx*dx + dy*dy + dz*dz);
    const double tmp  = sqrt(scale*scale + dist*dist);
    const double tmp3 = cuda_crlibm_pow_rn(tmp, 3.0);
    const double c    = mass / tmp3;
    *ax += dx * c;
    *ay += dy * c;
    *az += dz * c;
}

__global__ void nbCUDAExternalPotentialKernel(
    const double* __restrict__ d_posX,
    const double* __restrict__ d_posY,
    const double* __restrict__ d_posZ,
    double* __restrict__ d_accX,
    double* __restrict__ d_accY,
    double* __restrict__ d_accZ,
    int    nbody,
    /* External potential params packed by host. */
    int    sphereType, double sphereMass, double sphereScale,
    int    diskType,   double diskMass,   double diskScaleLength, double diskScaleHeight,
    int    disk2Type,  double disk2Mass,  double disk2ScaleLength, double disk2ScaleHeight,
    int    haloType,   double haloVHalo,  double haloScaleLength,  double haloFlattenZ,
    double haloMass,   /* used by NFWMass halo only */
    /* LMC params. */
    int    lmcType,    double lmcMass,    double lmcScale,
    double lmcPosX,    double lmcPosY,    double lmcPosZ,
    int    skipLMC)
{
    const unsigned int inc = blockDim.x * gridDim.x;
    for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
         (int) i < nbody;
         i += inc)
    {
        const double px = d_posX[i];
        const double py = d_posY[i];
        const double pz = d_posZ[i];
        /* Match CPU's nbExtAcceleration r-clamp (nbody_potential.c:680):
         *   real limit = mw_pow(2.0,-8.0);
         *   r = (|pos| <= limit) * limit + (|pos| > limit) * |pos|
         * which clamps r to >= 2^-8 for bodies near origin. The
         * earlier comment claimed to match this but the code wasn't
         * actually clamping — a v70 fix. Same branchless form as CPU
         * to bit-match. */
        const double absPos = sqrt(NBODY_CUDA_SQR(px) + NBODY_CUDA_SQR(py) + NBODY_CUDA_SQR(pz));
        const double limit  = 0.00390625;  /* 2^-8, exact in FP */
        const double r = (double)(absPos <= limit) * limit
                       + (double)(absPos >  limit) * absPos;

        double ax = 0.0, ay = 0.0, az = 0.0;

        /* IMPORTANT: order of accumulation matches CPU's
         * nbExtAcceleration EXACTLY: Disk → Disk2 → Halo → Bulge →
         * LMC. Reordering changes per-body force at ULP level which
         * compounds over 3418 steps and shifts BetaAvg/VelAvg
         * histogram bin assignments. */

        /* Primary disk. */
        switch (diskType)
        {
            case 1: /* Miyamoto-Nagai */
                nbCUDAAccelMiyamotoNagaiDisk(px, py, pz, diskMass,
                                              diskScaleLength, diskScaleHeight,
                                              &ax, &ay, &az);
                break;
            default: break;       /* other disk types: TODO */
        }

        /* Optional secondary disk. */
        if (disk2Type == 1)       /* Miyamoto-Nagai */
        {
            nbCUDAAccelMiyamotoNagaiDisk(px, py, pz, disk2Mass,
                                          disk2ScaleLength, disk2ScaleHeight,
                                          &ax, &ay, &az);
        }

        /* Halo. */
        switch (haloType)
        {
            case 1: /* Logarithmic */
                nbCUDAAccelLogHalo(px, py, pz, haloVHalo, haloScaleLength, haloFlattenZ,
                                    &ax, &ay, &az);
                break;
            case 2: /* NFW */
                nbCUDAAccelNFWHalo(px, py, pz, r, haloVHalo, haloScaleLength,
                                    &ax, &ay, &az);
                break;
            case 4: /* NFWMass — see NBodyCUDAHaloType / nbCUDAAccelNFWMassHalo */
                nbCUDAAccelNFWMassHalo(px, py, pz, r, haloMass, haloScaleLength,
                                        &ax, &ay, &az);
                break;
            case 5: /* SphericalNFWerkal — Erkal-variant spherical NFW */
                nbCUDAAccelSphericalNFWerkalHalo(px, py, pz, r, haloMass, haloScaleLength,
                                                  &ax, &ay, &az);
                break;
            default: break;       /* triaxial, AS, WE etc. — TODO */
        }

        /* Bulge (spherical) — added AFTER disk+halo to match CPU. */
        switch (sphereType)
        {
            case 1: /* Hernquist */
                nbCUDAAccelHernquistSphere(px, py, pz, r, sphereMass, sphereScale,
                                            &ax, &ay, &az);
                break;
            case 2: /* Plummer */
                nbCUDAAccelPlummerSphere(px, py, pz, r, sphereMass, sphereScale,
                                          &ax, &ay, &az);
                break;
            default: break;
        }

        /* LMC moving point-mass (last per CPU order).
         * Skipped on the FULL_KICK intermediate-correction step to
         * match the OpenCL `if (branch != -125.0)` guard. lmcScale2
         * is unused by the Plummer formula; threaded through for
         * future Hernquist-cutoff variants. */
        if (!skipLMC)
        {
            switch (lmcType)
            {
                case 1: /* Plummer */
                    nbCUDAAccelPlummerLMC(px, py, pz,
                                          lmcPosX, lmcPosY, lmcPosZ,
                                          lmcMass, lmcScale,
                                          &ax, &ay, &az);
                    break;
                default: break;   /* Hernquist / cutoff variants TODO */
            }
        }

        d_accX[i] += ax;
        d_accY[i] += ay;
        d_accZ[i] += az;
    }
}

extern "C" NBodyStatus_int nbCUDALaunchExternalPotential(
    struct NBodyCUDABuffers* buffers,
    int nbody,
    const NBodyCUDAExternalPotential* pot,
    NBodyCUDALMCType lmcType,
    double lmcMass,
    double lmcScale,
    double lmcScale2,
    double lmcPosX,
    double lmcPosY,
    double lmcPosZ,
    int skipLMC)
{
    if (!buffers || nbody != buffers->nbody) return NBODY_CUDA_ERROR;
    (void) lmcScale2;  /* reserved for Hernquist-cutoff LMC, not yet implemented */

    /* Default: no external potential, kernel is a no-op. */
    NBodyCUDAExternalPotential zero;
    if (!pot) { memset(&zero, 0, sizeof(zero)); pot = &zero; }

    const int block = NBODY_CUDA_BLOCK;
    const int grid  = (nbody + block - 1) / block;

    nbCUDAExternalPotentialKernel<<<grid, block>>>(
        buffers->d_posX, buffers->d_posY, buffers->d_posZ,
        buffers->d_accX, buffers->d_accY, buffers->d_accZ,
        nbody,
        pot->sphereType, pot->sphereMass, pot->sphereScale,
        pot->diskType,   pot->diskMass,   pot->diskScaleLength, pot->diskScaleHeight,
        pot->disk2Type,  pot->disk2Mass,  pot->disk2ScaleLength, pot->disk2ScaleHeight,
        pot->haloType,   pot->haloVHalo,  pot->haloScaleLength,  pot->haloFlattenZ,
        pot->haloMass,

        (int) lmcType, lmcMass, lmcScale,
        lmcPosX, lmcPosY, lmcPosZ,
        skipLMC);

    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess)
    {
        fprintf(stderr, "[nbody_cuda] externalPotential launch: %s\n", cudaGetErrorString(e));
        return NBODY_CUDA_ERROR;
    }
    return NBODY_CUDA_SUCCESS;
}

#undef NBODY_CUDA_SQR

/* ----- Phase 2: leapfrog integration kernel ----- */

/* Branch selector constants (kFullKickBranch / kSecondHalfBranch) are
 * defined just before the Phase 5 force-tree kernel above so both
 * kernels can share them. Match the OpenCL kernel's magic-number
 * encoding (kernels/nbody_kernels.cl:2369-2422). */

/* Leapfrog integration kernel. One body per thread, strided by
 * gridDim.x*blockDim.x so the same kernel works for nbody >> launched
 * threads if a future caller wants to cap occupancy. Mirrors
 * nbody_kernels.cl:2366-2428 line-for-line in arithmetic. */
__global__ void nbCUDAIntegrationKernel(double* __restrict__ d_posX,
                                        double* __restrict__ d_posY,
                                        double* __restrict__ d_posZ,
                                        double* __restrict__ d_velX,
                                        double* __restrict__ d_velY,
                                        double* __restrict__ d_velZ,
                                        const double* __restrict__ d_accX,
                                        const double* __restrict__ d_accY,
                                        const double* __restrict__ d_accZ,
                                        const int    nbody,
                                        const double timestep,
                                        const double branch,
                                        const double lmcAccX,
                                        const double lmcAccY,
                                        const double lmcAccZ)
{
    const unsigned int inc = blockDim.x * gridDim.x;

    for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
         (int) i < nbody;
         i += inc)
    {
        double px = d_posX[i];
        double py = d_posY[i];
        double pz = d_posZ[i];

        double ax = d_accX[i];
        double ay = d_accY[i];
        double az = d_accZ[i];

        /* LMC frame-acceleration adjustment: the first half-step in the
         * KDK pair adds it; the FULL_KICK case (interior re-kick) does not. */
        if (branch != kFullKickBranch)
        {
            ax += lmcAccX;
            ay += lmcAccY;
            az += lmcAccZ;
        }

        double vx = d_velX[i];
        double vy = d_velY[i];
        double vz = d_velZ[i];

        const double dvx = (0.5 * timestep) * ax;
        const double dvy = (0.5 * timestep) * ay;
        const double dvz = (0.5 * timestep) * az;

        vx += dvx;
        vy += dvy;
        vz += dvz;

        /* Separate ops to bit-match CPU's bodyAdvancePos:
         *   dr = mw_mulvs(Vel(p), dt);  mw_incaddv(Pos(p), dr);
         * which is `px += timestep * vx` (mulsd + addsd). */
        px += timestep * vx;
        py += timestep * vy;
        pz += timestep * vz;

        if (branch == kFullKickBranch)
        {
            vx += dvx;
            vy += dvy;
            vz += dvz;
        }

        if (branch == kFullKickBranch || branch != kSecondHalfBranch)
        {
            d_posX[i] = px;
            d_posY[i] = py;
            d_posZ[i] = pz;
        }

        d_velX[i] = vx;
        d_velY[i] = vy;
        d_velZ[i] = vz;
    }
}

extern "C" NBodyStatus_int nbCUDALaunchIntegration(struct NBodyCUDABuffers* buffers,
                                                   double timestep,
                                                   int nbody,
                                                   NBodyCUDAIntegrationPhase phase,
                                                   double lmcAccX,
                                                   double lmcAccY,
                                                   double lmcAccZ)
{
    if (!buffers || nbody != buffers->nbody || nbody <= 0)
    {
        return NBODY_CUDA_ERROR;
    }

    const int block = NBODY_CUDA_BLOCK;
    const int grid  = (nbody + block - 1) / block;
    const double branch = (double) (int) phase;  /* preserves the magic encoding */

    nbCUDAIntegrationKernel<<<grid, block>>>(buffers->d_posX,
                                             buffers->d_posY,
                                             buffers->d_posZ,
                                             buffers->d_velX,
                                             buffers->d_velY,
                                             buffers->d_velZ,
                                             buffers->d_accX,
                                             buffers->d_accY,
                                             buffers->d_accZ,
                                             nbody,
                                             timestep,
                                             branch,
                                             lmcAccX,
                                             lmcAccY,
                                             lmcAccZ);

    /* Catch launch-time errors (bad config, OOM, etc.). */
    cudaError_t launchErr = cudaGetLastError();
    if (launchErr != cudaSuccess)
    {
        fprintf(stderr, "[nbody_cuda] integration kernel launch failed: %s\n",
                cudaGetErrorString(launchErr));
        return NBODY_CUDA_ERROR;
    }

    /* Synchronous semantics: callers expect the device pos/vel to be
     * coherent on return. Asynchronous launch + later sync is a Phase 6
     * optimization. */
    return NBODY_CUDA_SUCCESS;
}

#endif /* NBODY_CUDA */
