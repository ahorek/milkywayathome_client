/*
 * C-callable interface to the CUDA device buffers used by the nbody port.
 *
 * The .cu translation unit can't safely include nbody_types.h /
 * milkyway_math.h (the milkyway/openpa header chain triggers nvcc
 * codegen bugs), so the boundary between the two sides is this small
 * POD-only API. Plain-C marshalling code in nbody_cuda_host.c packs
 * the AoS Body[] array into SoA arrays of doubles and hands them to
 * the .cu side through these entry points.
 *
 * On the device, all body data lives in struct-of-arrays layout
 * (pos_x, pos_y, pos_z, ...). This is what the existing OpenCL kernels
 * already use, and it's the layout the CUDA kernels in later phases
 * will read from for warp-coalesced loads.
 */

#ifndef _NBODY_CUDA_BUFFERS_H_
#define _NBODY_CUDA_BUFFERS_H_

#include "nbody_config.h"
#include "nbody_cuda.h"   /* NBodyStatus_int and NBodyCUDABuffers fwd-decl */

#ifdef __cplusplus
extern "C" {
#endif

#if NBODY_CUDA

/* Allocate device-side SoA buffers. On success, *outBuffers is set to
 * a heap-allocated wrapper holding the device pointers. Returns 0 on
 * success, non-zero on any cuda* failure (rolled back).
 *
 *   nbody - number of real bodies (sizes vel/acc).
 *   nNode - tree-node capacity for tree-code mode, or 0 for exact mode.
 *           When nNode > 0, pos and masses are sized to (nNode+1) so
 *           tree cells (which live at indices [nbody, nNode]) have
 *           storage too. The integration kernel only writes back to
 *           indices [0, nbody), so the cell-side entries get reused
 *           across steps by the tree construction kernels.
 *
 * For exact (N²) force computation pass nNode=0. For tree-code, pass
 * the value computed via nbCUDABuffersComputeNNode(). Then call
 * nbCUDATreeBuffersAlloc to attach the supporting tree-side scratch
 * (start, count, child, etc.). */
NBodyStatus_int nbCUDABuffersAlloc(struct NBodyCUDABuffers** outBuffers,
                                   int nbody,
                                   int nNode);

/* Allocate tree-storage buffers and attach them to an existing
 * buffer pack. Must be called after nbCUDABuffersAlloc.
 *
 *   nNode    - max tree nodes (typically 2*nbody, padded by warpSize).
 *              Caller computes via nbCUDABuffersComputeNNode().
 *   numSMs   - device SM count (used to size per-SM bounding-box
 *              reduction scratch). Get with cudaGetDeviceProperties.
 *   useQuad  - allocate quadrupole moment buffers as well.
 */
NBodyStatus_int nbCUDATreeBuffersAlloc(struct NBodyCUDABuffers* buffers,
                                       int nNode,
                                       int numSMs,
                                       int useQuad);

/* Compute the matching nNode size for a given nbody and device SM
 * count, mirroring the OpenCL nbFindNNode formula:
 *   nNode = max(2*nbody, 1024*numSMs), rounded up to warpSize (32). */
int nbCUDABuffersComputeNNode(int nbody, int numSMs);

/* Read the SM count from the currently-selected CUDA device.
 * Returns 0 on success; outSMs is populated. -1 on any cuda* error
 * (e.g. no CUDA device available). */
int nbCUDAGetDeviceSMCount(int* outSMs);

/* Accessors for the opaque NBodyCUDABuffers — host C code can't
 * inspect the struct fields directly. */
int nbCUDABuffersGetNbody(const struct NBodyCUDABuffers* buffers);
int nbCUDABuffersGetNNode(const struct NBodyCUDABuffers* buffers);
/* Returns the cumulative max tree depth recorded in d_treeStatus, or -1 on error. */
int nbCUDABuffersGetMaxDepth(const struct NBodyCUDABuffers* buffers);

/* DEBUG: dump tree-cell CoMs/Rcrit2 in DFS order. */
int nbCUDABuffersDumpTree(const struct NBodyCUDABuffers* buffers,
                          int nbody, int nNode, const char* path);

/* DEBUG: dump tree-cell CoM+Rcrit2+quadXX..ZZ (11 doubles/cell). */
int nbCUDABuffersDumpTreeQuad(const struct NBodyCUDABuffers* buffers,
                              int nbody, int nNode, const char* path);

/* Free all device buffers (body + tree if allocated) and the wrapper
 * struct. Safe to call with NULL. */
void nbCUDABuffersFree(struct NBodyCUDABuffers* buffers);

/* Upload SoA host arrays of doubles to the device buffers. Each host
 * array must be `nbody` doubles long. The masses array must have been
 * uploaded once (the CPU code never mutates body masses during a
 * simulation); pos/vel get re-uploaded any time the host view is
 * authoritative. */
NBodyStatus_int nbCUDABuffersUploadBodies(struct NBodyCUDABuffers* buffers,
                                          const double* hPosX,
                                          const double* hPosY,
                                          const double* hPosZ,
                                          const double* hVelX,
                                          const double* hVelY,
                                          const double* hVelZ,
                                          const double* hMasses,
                                          int nbody);

/* Download pos/vel from the device into host SoA arrays. Used on
 * checkpoint and at end-of-simulation to make the CPU view
 * authoritative again. */
NBodyStatus_int nbCUDABuffersDownloadBodies(const struct NBodyCUDABuffers* buffers,
                                            double* hPosX,
                                            double* hPosY,
                                            double* hPosZ,
                                            double* hVelX,
                                            double* hVelY,
                                            double* hVelZ,
                                            int nbody);

/* Upload only the acceleration arrays to the device. Used in Phase 2
 * where forces are still being computed on the CPU and only the
 * integrator runs on GPU. */
NBodyStatus_int nbCUDABuffersUploadAccels(struct NBodyCUDABuffers* buffers,
                                          const double* hAccX,
                                          const double* hAccY,
                                          const double* hAccZ,
                                          int nbody);

/* Download acceleration arrays. For diagnostics. */
NBodyStatus_int nbCUDABuffersDownloadAccels(const struct NBodyCUDABuffers* buffers,
                                            double* hAccX,
                                            double* hAccY,
                                            double* hAccZ,
                                            int nbody);

/* Encoding of the leapfrog phase the integration kernel is running.
 * Mirrors the OpenCL `LMCbranching` constant so the same arithmetic
 * branches inside the kernel select the same code path:
 *   FIRST_HALF  -> KDK first half-kick + drift; LMC accel adjustment applied.
 *   SECOND_HALF -> KDK second half-kick (no drift, no pos write).
 *   FULL_KICK   -> Combined K+K with drift, no LMC adjustment (interior step).
 * The wire values match the OpenCL kernel's magic numbers exactly. */
typedef enum {
    NBODY_CUDA_INT_FIRST_HALF  = 0,           /* arbitrary non-magic value */
    NBODY_CUDA_INT_SECOND_HALF = -1024,       /* matches LMCbranching code */
    NBODY_CUDA_INT_FULL_KICK   = -125         /* matches LMCbranching code */
} NBodyCUDAIntegrationPhase;

/* Launch the direct N-squared gravity force kernel. Computes the
 * gravitational acceleration on every body from every other body,
 * with `eps2` softening, and writes the result into the d_acc*
 * buffers (overwriting any previous value).
 *
 * This is the GPU analogue of nbGravity_Exact() in nbody_grav.c —
 * mirrors the OpenCL forceCalculation_Exact kernel, minus the
 * external Milky Way / LMC potential (those land in a follow-up
 * phase). Tile-based shared-memory blocking matches the OpenCL
 * pattern for warp-coalesced loads.
 *
 * Synchronous: blocks until the kernel completes.
 */
NBodyStatus_int nbCUDALaunchForceExact(struct NBodyCUDABuffers* buffers,
                                       int nbody,
                                       double eps2);

/* ----- Phase 4: Barnes-Hut tree construction launchers ----- */

/* Compute the bounding box that contains all bodies. Writes result
 * into the tree's root cell (last node, index nNode) and resets the
 * root's mass/child pointers. After this, posX[nNode]..posZ[nNode]
 * hold the box centre and treeStatus->radius holds the half-side. */
NBodyStatus_int nbCUDALaunchBoundingBox(struct NBodyCUDABuffers* buffers,
                                        int nbody,
                                        int nNode);

/* Reset child pointers and node fields ahead of buildTree. Must run
 * before nbCUDALaunchBuildTree. */
NBodyStatus_int nbCUDALaunchBuildTreeClear(struct NBodyCUDABuffers* buffers,
                                           int nNode);

/* Insert each body into the Barnes-Hut octree. Atomic CAS on child
 * pointers serializes inserts at contended cells. */
NBodyStatus_int nbCUDALaunchBuildTree(struct NBodyCUDABuffers* buffers,
                                      int nbody,
                                      int nNode);

/* Reset cell mass-rest counters before summarization. */
NBodyStatus_int nbCUDALaunchSummarizationClear(struct NBodyCUDABuffers* buffers,
                                               int nbody,
                                               int nNode);

/* Bottom-up tree summarization: compute centre-of-mass and total
 * mass for each cell from its children. */
NBodyStatus_int nbCUDALaunchSummarization(struct NBodyCUDABuffers* buffers,
                                          int nbody,
                                          int nNode);

/* Reorder body indices in the `sort` array so that bodies in the
 * same tree leaf are contiguous. Improves coalescing in the
 * subsequent force kernel. */
NBodyStatus_int nbCUDALaunchSort(struct NBodyCUDABuffers* buffers,
                                 int nbody,
                                 int nNode);

/* Compute quadrupole moments for each cell. Only meaningful when
 * useQuad was true at tree-buffer alloc time. */
NBodyStatus_int nbCUDALaunchQuadMoments(struct NBodyCUDABuffers* buffers,
                                        int nNode);

/* ----- Phase 5b: external Milky Way + LMC potential ----- */

/* Disk model selector. Wire values match the disk_t enum in
 * nbody_potential_types.h. */
typedef enum {
    NBODY_CUDA_DISK_NONE              = 0,
    NBODY_CUDA_DISK_MIYAMOTO_NAGAI    = 1,
    NBODY_CUDA_DISK_FREEMAN           = 2,
    NBODY_CUDA_DISK_DOUBLE_EXP        = 3,
    NBODY_CUDA_DISK_SECH_EXP          = 4,
    NBODY_CUDA_DISK_BAR               = 5
} NBodyCUDADiskType;

/* Spherical (bulge) selector. Wire values match spherical_t. */
typedef enum {
    NBODY_CUDA_SPHERE_NONE      = 0,
    NBODY_CUDA_SPHERE_HERNQUIST = 1,
    NBODY_CUDA_SPHERE_PLUMMER   = 2
} NBodyCUDASphericalType;

/* Halo selector. Compact internal IDs (NOT the halo_t enum values);
 * the CPU halo_t value is mapped to one of these in
 * nbCUDAPackPotential before being passed into the kernel. */
typedef enum {
    NBODY_CUDA_HALO_NONE                = 0,
    NBODY_CUDA_HALO_LOG                 = 1,
    NBODY_CUDA_HALO_NFW                 = 2,
    NBODY_CUDA_HALO_TRIAXIAL            = 3,
    NBODY_CUDA_HALO_NFWMASS             = 4,  /* mass-parametrized NFW (Halo.nfwmass{...}) */
    NBODY_CUDA_HALO_SPHERICAL_NFW_ERKAL = 5   /* Erkal-variant spherical NFW (Halo.sphericalnfwerkal{...}) */
    /* Other halo types fall back to CPU path until ported. */
} NBodyCUDAHaloType;

/* LMC point-mass model selector. Wire values match the OpenCL
 * LMC_function constants in the Lua config. */
typedef enum {
    NBODY_CUDA_LMC_NONE              = 0,
    NBODY_CUDA_LMC_PLUMMER           = 1,
    NBODY_CUDA_LMC_HERNQUIST         = 2,
    NBODY_CUDA_LMC_HERNQUIST_CUTOFF  = 3
} NBodyCUDALMCType;

/* POD struct holding all external-potential parameters. Passed by
 * value into the external potential kernel; the host packs the
 * matching CPU Potential struct fields into one of these before each
 * launch. Fields not relevant to the selected model type are ignored. */
typedef struct NBodyCUDAExternalPotential
{
    int    sphereType;        /* NBodyCUDASphericalType */
    double sphereMass;
    double sphereScale;

    int    diskType;          /* NBodyCUDADiskType */
    double diskMass;
    double diskScaleLength;
    double diskScaleHeight;

    int    disk2Type;         /* second disk component (often zero) */
    double disk2Mass;
    double disk2ScaleLength;
    double disk2ScaleHeight;

    int    haloType;          /* NBodyCUDAHaloType */
    double haloVHalo;
    double haloScaleLength;
    double haloFlattenZ;
    double haloMass;          /* used by NFW/etc */
} NBodyCUDAExternalPotential;

/* Launch the external potential evaluation kernel. Adds the
 * Milky Way (bulge + disk(s) + halo) plus optional moving Plummer
 * LMC contribution into d_acc. Run after the tree-walk force kernel
 * and before the integration kernel.
 *
 *   pot               - host-side parameter pack (passed by value)
 *   lmcType           - which LMC formula to use; pass NONE to skip
 *   lmcMass / lmcScale / lmcScale2 - LMC point-mass parameters
 *   lmcPosX/Y/Z       - LMC current position (per-step)
 *   skipLMC           - if non-zero, skip the LMC contribution this
 *                       step (matches OpenCL `if (branch != FULL_KICK)`).
 */
NBodyStatus_int nbCUDALaunchExternalPotential(struct NBodyCUDABuffers* buffers,
                                              int nbody,
                                              const NBodyCUDAExternalPotential* pot,
                                              NBodyCUDALMCType lmcType,
                                              double lmcMass,
                                              double lmcScale,
                                              double lmcScale2,
                                              double lmcPosX,
                                              double lmcPosY,
                                              double lmcPosZ,
                                              int skipLMC);

/* ----- Phase 5: Barnes-Hut tree-walk force kernel ----- */

/* Walk the BH tree for every body and accumulate the gravity force
 * with `eps2` softening, theta opening criterion, and (optionally)
 * quadrupole-moment corrections. Mirrors the OpenCL forceCalculation
 * kernel (nbody_kernels.cl:1902-2253).
 *
 * External Milky Way / LMC potential is NOT applied by this kernel —
 * it runs as a separate add-on pass (Phase 5b). The kernel still
 * applies the per-body intermediate-kick velocity correction
 * (`branch == FULL_KICK` path) using the old-vs-new acceleration
 * delta.
 *
 *   eps2    - squared softening length
 *   theta   - opening criterion (BH opening angle); 0.0 means "always open"
 *   updateVel - 1 to write the corrected velocity in the FULL_KICK
 *               branch, 0 to skip the writeback (matches OpenCL).
 *   phase   - leapfrog phase encoding (same enum as integration kernel)
 *
 * Synchronous: blocks until the kernel completes.
 */
NBodyStatus_int nbCUDALaunchForceTree(struct NBodyCUDABuffers* buffers,
                                      int nbody,
                                      int nNode,
                                      double eps2,
                                      double theta,
                                      int useQuad,
                                      int updateVel,
                                      double timestep,
                                      NBodyCUDAIntegrationPhase phase);

/* Launch the leapfrog integration kernel on the bodies in `buffers`.
 *
 * timestep        - simulation dt (seconds in code's natural units).
 * nbody           - number of bodies (must equal buffers->nbody).
 * phase           - which leapfrog phase to apply (see enum above).
 * lmcAccX/Y/Z     - per-step LMC frame-acceleration correction. Only
 *                   applied when phase == FIRST_HALF (matches OpenCL
 *                   `if (branch != -125.0) acc += LMCacci`). Pass 0,0,0
 *                   when LMC isn't active.
 *
 * Synchronous: blocks until the kernel completes (cudaDeviceSynchronize).
 */
NBodyStatus_int nbCUDALaunchIntegration(struct NBodyCUDABuffers* buffers,
                                        double timestep,
                                        int nbody,
                                        NBodyCUDAIntegrationPhase phase,
                                        double lmcAccX,
                                        double lmcAccY,
                                        double lmcAccZ);

#endif /* NBODY_CUDA */

#ifdef __cplusplus
}
#endif

#endif /* _NBODY_CUDA_BUFFERS_H_ */
