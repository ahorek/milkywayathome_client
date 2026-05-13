/*
 * Plain-C marshalling layer between the nbody runtime (which holds
 * bodies in AoS Body[] arrays) and the CUDA backend (which holds
 * bodies in SoA double[] arrays on the device).
 *
 * This file is the only place where nbody_types.h, milkyway_math.h,
 * and the .cu translation unit see each other indirectly. By packing
 * AoS -> SoA on the C side and handing pure double-pointer arrays into
 * nbody_cuda.cu, we keep nvcc away from the milkyway/openpa header
 * chain that triggers nvcc segfaults.
 */

#include "nbody_config.h"

#if NBODY_CUDA

/* Many switch statements below intentionally only enumerate the
 * potential model types the CUDA backend supports — the rest fall
 * through to the default branch (which signals an unsupported
 * potential, sending the runtime back to the CPU path). The
 * project's default warning set treats unhandled enum values as
 * errors; silence that locally rather than enumerating ~30 enum
 * values per switch. */
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wswitch-enum"

#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "nbody_types.h"
#include "milkyway_math.h"
#include "milkyway_util.h"
#include "nbody_cuda.h"
#include "nbody_cuda_buffers.h"

/* Allocate a contiguous block of 7*nbody doubles to hold the SoA pack
 * (pos x/y/z, vel x/y/z, mass) used during a single round-trip. One
 * allocation keeps the cost down and the allocations aligned. */
static double* nbCUDAAllocSoAScratch(int nbody)
{
    return (double*) mwCalloc((size_t) 7 * (size_t) nbody, sizeof(double));
}

/* Marshal `nbody` AoS bodies into the SoA scratch buffer. Layout in
 * scratch: [posX][posY][posZ][velX][velY][velZ][mass], each block
 * `nbody` doubles. */
static void nbCUDAPackBodiesToSoA(const Body* bodies, int nbody, double* scratch)
{
    double* px = scratch + 0 * (size_t) nbody;
    double* py = scratch + 1 * (size_t) nbody;
    double* pz = scratch + 2 * (size_t) nbody;
    double* vx = scratch + 3 * (size_t) nbody;
    double* vy = scratch + 4 * (size_t) nbody;
    double* vz = scratch + 5 * (size_t) nbody;
    double* m  = scratch + 6 * (size_t) nbody;

    for (int i = 0; i < nbody; ++i)
    {
        const Body* b = &bodies[i];
        px[i] = X(Pos(b));
        py[i] = Y(Pos(b));
        pz[i] = Z(Pos(b));
        vx[i] = X(Vel(b));
        vy[i] = Y(Vel(b));
        vz[i] = Z(Vel(b));
        m[i]  = Mass(b);
    }
}

/* Inverse of nbCUDAPackBodiesToSoA: copy SoA pos/vel back into AoS
 * bodies. Mass is not written because the device never mutates it. */
static void nbCUDAUnpackSoAToBodies(const double* scratch, Body* bodies, int nbody)
{
    const double* px = scratch + 0 * (size_t) nbody;
    const double* py = scratch + 1 * (size_t) nbody;
    const double* pz = scratch + 2 * (size_t) nbody;
    const double* vx = scratch + 3 * (size_t) nbody;
    const double* vy = scratch + 4 * (size_t) nbody;
    const double* vz = scratch + 5 * (size_t) nbody;

    for (int i = 0; i < nbody; ++i)
    {
        Body* b = &bodies[i];
        X(Pos(b)) = px[i];
        Y(Pos(b)) = py[i];
        Z(Pos(b)) = pz[i];
        X(Vel(b)) = vx[i];
        Y(Vel(b)) = vy[i];
        Z(Vel(b)) = vz[i];
    }
}

/* Public entry: allocate device buffers sized for st->nbody bodies +
 * (nNode+1 - nbody) tree cells, and upload the current st->bodytab. */
NBodyStatus nbCUDAMarshalBodiesToDevice(NBodyState* st, int nNode)
{
    if (!st || st->nbody <= 0) return NBODY_ERROR;

    /* Allocate device buffers if the caller hasn't already. */
    if (!st->cudaBuffers)
    {
        if (nbCUDABuffersAlloc(&st->cudaBuffers, st->nbody, nNode) != 0)
        {
            return NBODY_ERROR;
        }
    }

    double* scratch = nbCUDAAllocSoAScratch(st->nbody);
    if (!scratch) return NBODY_ERROR;

    nbCUDAPackBodiesToSoA(st->bodytab, st->nbody, scratch);

    int rc = nbCUDABuffersUploadBodies(st->cudaBuffers,
                                       scratch + 0 * (size_t) st->nbody,
                                       scratch + 1 * (size_t) st->nbody,
                                       scratch + 2 * (size_t) st->nbody,
                                       scratch + 3 * (size_t) st->nbody,
                                       scratch + 4 * (size_t) st->nbody,
                                       scratch + 5 * (size_t) st->nbody,
                                       scratch + 6 * (size_t) st->nbody,
                                       st->nbody);
    free(scratch);
    return (rc == 0) ? NBODY_SUCCESS : NBODY_ERROR;
}

/* Public entry: download pos/vel from device back into st->bodytab.
 * Mass is not touched. */
NBodyStatus nbCUDAMarshalBodiesFromDevice(NBodyState* st)
{
    if (!st || st->nbody <= 0 || !st->cudaBuffers) return NBODY_ERROR;

    double* scratch = nbCUDAAllocSoAScratch(st->nbody);
    if (!scratch) return NBODY_ERROR;

    int rc = nbCUDABuffersDownloadBodies(st->cudaBuffers,
                                         scratch + 0 * (size_t) st->nbody,
                                         scratch + 1 * (size_t) st->nbody,
                                         scratch + 2 * (size_t) st->nbody,
                                         scratch + 3 * (size_t) st->nbody,
                                         scratch + 4 * (size_t) st->nbody,
                                         scratch + 5 * (size_t) st->nbody,
                                         st->nbody);
    if (rc == 0)
    {
        nbCUDAUnpackSoAToBodies(scratch, st->bodytab, st->nbody);
    }
    free(scratch);
    return (rc == 0) ? NBODY_SUCCESS : NBODY_ERROR;
}

/* Public entry: free device buffers and clear the handle on st. */
void nbCUDAReleaseBodyBuffers(NBodyState* st)
{
    if (!st) return;
    if (st->cudaBuffers)
    {
        nbCUDABuffersFree(st->cudaBuffers);
        st->cudaBuffers = NULL;
    }
}

/* ----- Phase 5c: lifecycle + orchestration -----
 * nbInitCUDA / nbReleaseCUDA / nbStepSystemCUDA / nbRunSystemCUDA
 * implement the entry points declared in nbody_cuda.h. They live in
 * the C wrapper (not nbody_cuda.cu) because they need to inspect
 * NBodyState / NBodyCtx fields — the .cu translation unit can't
 * safely include nbody_types.h.
 */

#include "nbody_potential_types.h"
#include "nbody_checkpoint.h"
#include "milkyway_boinc_util.h"
#include "nbody_potential.h"  /* nbExtAcceleration */
#include "nbody_friction.h"   /* dynamicalFriction_LMC */
#include "nbody_orbit_integrator.h"  /* getLMCArray, getLMCPosVel */
#include "nbody.h"                  /* NBodyFlags */
#include "nbody_plain.h"            /* nbGetLikelihoodForBest */
#include "nbody_util.h"             /* nbCenterOfMass */

/* Map ctx->pot (and whatever sub-type fields are set) onto the
 * POD CUDAExternalPotential param pack. Unsupported or "None"
 * sub-types map to NBODY_CUDA_*_NONE so the kernel skips them. */
static void nbCUDAPackPotential(const NBodyCtx* ctx, NBodyCUDAExternalPotential* out)
{
    memset(out, 0, sizeof(*out));

    /* Spherical (bulge). */
    switch (ctx->pot.sphere[0].type)
    {
        case HernquistSpherical:
            out->sphereType = NBODY_CUDA_SPHERE_HERNQUIST;
            out->sphereMass  = ctx->pot.sphere[0].mass;
            out->sphereScale = ctx->pot.sphere[0].scale;
            break;
        case PlummerSpherical:
            out->sphereType = NBODY_CUDA_SPHERE_PLUMMER;
            out->sphereMass  = ctx->pot.sphere[0].mass;
            out->sphereScale = ctx->pot.sphere[0].scale;
            break;
        default:
            out->sphereType = NBODY_CUDA_SPHERE_NONE;
            break;
    }

    /* Primary disk. */
    switch (ctx->pot.disk.type)
    {
        case MiyamotoNagaiDisk:
            out->diskType        = NBODY_CUDA_DISK_MIYAMOTO_NAGAI;
            out->diskMass        = ctx->pot.disk.mass;
            out->diskScaleLength = ctx->pot.disk.scaleLength;
            out->diskScaleHeight = ctx->pot.disk.scaleHeight;
            break;
        default:
            /* Freeman / DoubleExp / SechExp / Bar — not yet ported.
             * Caller should fall back to CPU path if any of these
             * are configured. */
            out->diskType = NBODY_CUDA_DISK_NONE;
            break;
    }

    /* Secondary disk (optional). */
    switch (ctx->pot.disk2.type)
    {
        case MiyamotoNagaiDisk:
            out->disk2Type        = NBODY_CUDA_DISK_MIYAMOTO_NAGAI;
            out->disk2Mass        = ctx->pot.disk2.mass;
            out->disk2ScaleLength = ctx->pot.disk2.scaleLength;
            out->disk2ScaleHeight = ctx->pot.disk2.scaleHeight;
            break;
        default:
            out->disk2Type = NBODY_CUDA_DISK_NONE;
            break;
    }

    /* Halo. */
    switch (ctx->pot.halo.type)
    {
        case LogarithmicHalo:
            out->haloType        = NBODY_CUDA_HALO_LOG;
            out->haloVHalo       = ctx->pot.halo.vhalo;
            out->haloScaleLength = ctx->pot.halo.scaleLength;
            out->haloFlattenZ    = ctx->pot.halo.flattenZ;
            break;
        case NFWHalo:
            out->haloType        = NBODY_CUDA_HALO_NFW;
            out->haloVHalo       = ctx->pot.halo.vhalo;
            out->haloScaleLength = ctx->pot.halo.scaleLength;
            break;
        case NFWMassHalo:
            out->haloType        = NBODY_CUDA_HALO_NFWMASS;
            out->haloMass        = ctx->pot.halo.mass;
            out->haloScaleLength = ctx->pot.halo.scaleLength;
            break;
        case SphericalNFWerkalHalo:
            out->haloType        = NBODY_CUDA_HALO_SPHERICAL_NFW_ERKAL;
            out->haloMass        = ctx->pot.halo.mass;
            out->haloScaleLength = ctx->pot.halo.scaleLength;
            break;
        default:
            /* Triaxial / AS / WE / Plummer / Hernquist / Ninkovic halos:
             * not yet ported. CPU fallback applies. */
            out->haloType = NBODY_CUDA_HALO_NONE;
            break;
    }
}

/* Returns TRUE if the configured potential models are all supported
 * by the CUDA backend. Anything unsupported pushes the runtime to
 * the CPU fallback path (with a warning printed at init time). */
/* The string-name helpers (nbCUDASphericalName / nbCUDADiskName /
 * nbCUDAHaloName) and nbPrintPotentialModel live in nbody_potential.c
 * so they can be called from the always-linked CPU path. Declared in
 * nbody_potential.h, transitively included via nbody.h. */

static int nbCUDACanHandlePotential(const NBodyCtx* ctx)
{
    /* Lua-driven custom potentials are not supported on GPU. */
    if (ctx->potentialType == EXTERNAL_POTENTIAL_CUSTOM_LUA)
    {
        mw_printf("[nbody_cuda] reject: custom Lua potential not supported on GPU\n");
        return 0;
    }
    /* No external potential — fine. */
    if (ctx->potentialType == EXTERNAL_POTENTIAL_NONE)
    {
        return 1;
    }
    /* Spherical: only Plummer/Hernquist/None. */
    if (ctx->pot.sphere[0].type != NoSpherical
     && ctx->pot.sphere[0].type != HernquistSpherical
     && ctx->pot.sphere[0].type != PlummerSpherical)
    {
        mw_printf("[nbody_cuda] reject: bulge type %s not supported on GPU\n",
                  nbCUDASphericalName(ctx->pot.sphere[0].type));
        return 0;
    }
    /* Primary disk: only Miyamoto-Nagai or None. */
    if (ctx->pot.disk.type != NoDisk
     && ctx->pot.disk.type != MiyamotoNagaiDisk)
    {
        mw_printf("[nbody_cuda] reject: primary disk type %s not supported on GPU\n",
                  nbCUDADiskName(ctx->pot.disk.type));
        return 0;
    }
    /* Secondary disk: same. */
    if (ctx->pot.disk2.type != NoDisk
     && ctx->pot.disk2.type != MiyamotoNagaiDisk)
    {
        mw_printf("[nbody_cuda] reject: secondary disk type %s not supported on GPU\n",
                  nbCUDADiskName(ctx->pot.disk2.type));
        return 0;
    }
    /* Halo: Log, NFW, NFWMass, or SphericalNFWerkal. */
    if (ctx->pot.halo.type != NoHalo
     && ctx->pot.halo.type != LogarithmicHalo
     && ctx->pot.halo.type != NFWHalo
     && ctx->pot.halo.type != NFWMassHalo
     && ctx->pot.halo.type != SphericalNFWerkalHalo)
    {
        mw_printf("[nbody_cuda] reject: halo type %s not supported on GPU\n",
                  nbCUDAHaloName(ctx->pot.halo.type));
        return 0;
    }
    return 1;
}

/* nbInitCUDA: probe the CUDA device, allocate body+tree buffers,
 * upload initial body state. Sets st->usesCUDA=TRUE on success. */
NBodyStatus_int nbInitCUDA(const NBodyCtx* ctx, NBodyState* st)
{
    if (!ctx || !st) return NBODY_ERROR;

    if (!st->bodytab || st->nbody <= 0)
    {
        mw_printf("[nbody_cuda] bodies not yet loaded — call nbInitCUDA after Lua setup\n");
        return NBODY_ERROR;
    }

    if (!nbCUDACanHandlePotential(ctx))
    {
        mw_printf("[nbody_cuda] potential model not supported by CUDA backend — falling back to CPU\n");
        return NBODY_ERROR;
    }

    #ifndef NBODY_CUDA_FORCE_EXACT
      #define NBODY_CUDA_FORCE_EXACT 0
    #endif
    if (ctx->criterion == Exact && !NBODY_CUDA_FORCE_EXACT)
    {
        mw_printf("[nbody_cuda] EXACT criterion not yet wired into CUDA path — use TreeCode\n");
        return NBODY_ERROR;
    }

    int numSMs = 0;
    if (nbCUDAGetDeviceSMCount(&numSMs) != 0)
    {
        mw_printf("[nbody_cuda] no CUDA device available\n");
        return NBODY_ERROR;
    }

    /* Pull the LMC pos/vel and shift array out of the static globals
     * populated by the reverse-orbit integration during setup. The CPU
     * and OpenCL paths do this at the top of their main loops; the
     * CUDA path was missing it, leaving st->LMCpos = (0,0,0) and
     * dynamicalFriction_LMC dividing by zero -> NaN cascade. */
    if (ctx->LMC && !st->shiftByLMC)
    {
        mwvector* shiftLMC = NULL;
        size_t sizeLMC = 0;
        mwvector LMCx = ZERO_VECTOR;
        mwvector LMCv = ZERO_VECTOR;
        getLMCArray(&shiftLMC, &sizeLMC);
        setLMCShiftArray(st, shiftLMC, sizeLMC);
        getLMCPosVel(&LMCx, &LMCv);
        setLMCPosVel(st, LMCx, LMCv);
        mw_printf("[nbody_cuda] LMC init: pos=(%.4g,%.4g,%.4g) vel=(%.4g,%.4g,%.4g) nShift=%zu\n",
                  X(st->LMCpos), Y(st->LMCpos), Z(st->LMCpos),
                  X(st->LMCvel), Y(st->LMCvel), Z(st->LMCvel),
                  sizeLMC);
    }

    const int useQuad = ctx->useQuad ? 1 : 0;

    /* Compute tree-node capacity FIRST so the body buffer alloc sizes
     * pos/mass to (nNode+1). Cells live at indices [nbody, nNode] in
     * those same buffers and are written by the tree-construction
     * kernels (boundingBox stores root cell position into d_posX[nNode]
     * etc.). With pos/mass sized only to `nbody` we'd write past end. */
    const int nNode = nbCUDABuffersComputeNNode(st->nbody, numSMs);

    /* On a calibration re-init (st->cudaBuffers already set), we just
     * re-upload the latest body state; the device pos/mass arrays are
     * already sized correctly and the tree-side scratch persists. */
    const int firstInit = (st->cudaBuffers == NULL);

    if (nbCUDAMarshalBodiesToDevice(st, nNode) != NBODY_SUCCESS)
    {
        mw_printf("[nbody_cuda] failed to upload bodies\n");
        return NBODY_ERROR;
    }

    if (firstInit)
    {
        if (nbCUDATreeBuffersAlloc(st->cudaBuffers, nNode, numSMs, useQuad) != 0)
        {
            mw_printf("[nbody_cuda] tree buffer alloc failed\n");
            nbCUDAReleaseBodyBuffers(st);
            return NBODY_ERROR;
        }
    }

    st->usesCUDA = TRUE;
    st->usesQuad = ctx->useQuad;
    st->usesExact = (ctx->criterion == Exact);

    /* Seed d_acc with the initial-step accelerations so the first
     * call to nbStepSystemCUDA's "first half-kick" reads a meaningful
     * value (otherwise OLD acc is zeros from cudaMalloc, causing a
     * silent first-step velocity drop). One full force pass covers
     * tree-walk gravity + external potential. No integration here. */
    NBodyCUDAExternalPotential pot;
    nbCUDAPackPotential(ctx, &pot);
    int initForceFailed = 0;
    #if NBODY_CUDA_FORCE_EXACT
        initForceFailed = (nbCUDALaunchForceExact(st->cudaBuffers, st->nbody, ctx->eps2) != 0)
            || (nbCUDALaunchExternalPotential(st->cudaBuffers, st->nbody, &pot,
                                              ctx->LMC ? NBODY_CUDA_LMC_PLUMMER : NBODY_CUDA_LMC_NONE,
                                              ctx->LMCmass, ctx->LMCscale, 0.0,
                                              X(st->LMCpos), Y(st->LMCpos), Z(st->LMCpos),
                                              /*skipLMC=*/0) != 0);
    #else
        initForceFailed =
               (nbCUDALaunchBoundingBox(st->cudaBuffers, st->nbody, nNode) != 0)
            || (nbCUDALaunchBuildTreeClear(st->cudaBuffers, nNode) != 0)
            || (nbCUDALaunchBuildTree(st->cudaBuffers, st->nbody, nNode) != 0)
            || (nbCUDALaunchSummarizationClear(st->cudaBuffers, st->nbody, nNode) != 0)
            || (nbCUDALaunchSummarization(st->cudaBuffers, st->nbody, nNode) != 0)
            || (nbCUDALaunchSort(st->cudaBuffers, st->nbody, nNode) != 0)
            || (useQuad && nbCUDALaunchQuadMoments(st->cudaBuffers, nNode) != 0)
            || (useQuad && nbCUDALaunchQuadPack(st->cudaBuffers, nNode) != 0)
            || (nbCUDALaunchForceTree(st->cudaBuffers, st->nbody, nNode,
                                      ctx->eps2, ctx->theta, useQuad,
                                      /*updateVel=*/0,
                                      ctx->timestep,
                                      NBODY_CUDA_INT_FIRST_HALF) != 0)
            || (nbCUDALaunchExternalPotential(st->cudaBuffers, st->nbody, &pot,
                                              ctx->LMC ? NBODY_CUDA_LMC_PLUMMER : NBODY_CUDA_LMC_NONE,
                                              ctx->LMCmass, ctx->LMCscale, 0.0,
                                              X(st->LMCpos), Y(st->LMCpos), Z(st->LMCpos),
                                              /*skipLMC=*/0) != 0);
    #endif
    if (initForceFailed)
    {
        mw_printf("[nbody_cuda] initial force pass failed\n");
        nbReleaseCUDA(st);
        return NBODY_ERROR;
    }

    mw_printf("[nbody_cuda] initialized: nbody=%d nNode=%d numSMs=%d useQuad=%d\n",
              st->nbody, nNode, numSMs, useQuad);

    /* DEBUG: dump per-body initial acc to a binary file when
     * NBODY_DUMP_ACC=1 in env. Mirrors the same dump in nbody_plain.c
     * so we can diff CPU vs CUDA at step 0. Exits after dump. */
    if (getenv("NBODY_DUMP_ACC"))
    {
        const char* path = getenv("NBODY_DUMP_ACC_FILE");
        if (!path) path = "/tmp/cuda_smoketest/cuda_acc.bin";
        int nb = st->nbody;
        double* ax = (double*) malloc(sizeof(double) * (size_t) nb);
        double* ay = (double*) malloc(sizeof(double) * (size_t) nb);
        double* az = (double*) malloc(sizeof(double) * (size_t) nb);
        if (ax && ay && az &&
            nbCUDABuffersDownloadAccels(st->cudaBuffers, ax, ay, az, nb) == 0)
        {
            FILE* f = fopen(path, "wb");
            if (f)
            {
                fwrite(&nb, sizeof(int), 1, f);
                for (int i = 0; i < nb; ++i)
                {
                    fwrite(&ax[i], sizeof(double), 1, f);
                    fwrite(&ay[i], sizeof(double), 1, f);
                    fwrite(&az[i], sizeof(double), 1, f);
                }
                fclose(f);
                mw_printf("[DEBUG] dumped %d body accs to %s\n", nb, path);
            }
        }
        free(ax); free(ay); free(az);
        exit(0);
    }

    /* DEBUG: dump CUDA-side tree CoM/mass/Rcrit2 in DFS order to
     * compare with CPU's tree dump. */
    if (getenv("NBODY_DUMP_TREE"))
    {
        const char* path = getenv("NBODY_DUMP_TREE_FILE");
        if (!path) path = "/tmp/v100_diag/cuda_tree.bin";
        nbCUDABuffersDumpTree(st->cudaBuffers, st->nbody, nNode, path);
        exit(0);
    }

    /* DEBUG: dump tree+quad. */
    if (getenv("NBODY_DUMP_TREEQUAD"))
    {
        const char* path = getenv("NBODY_DUMP_TREEQUAD_FILE");
        if (!path) path = "/tmp/v100_diag/cuda_treeq.bin";
        nbCUDABuffersDumpTreeQuad(st->cudaBuffers, st->nbody, nNode, path);
        exit(0);
    }

    /* DEBUG: dump CUDA body positions for CPU/CUDA comparison. */
    if (getenv("NBODY_DUMP_POS"))
    {
        const char* path = getenv("NBODY_DUMP_POS_FILE");
        if (!path) path = "/tmp/v100_diag/cuda_pos.bin";
        int nb = st->nbody;
        double* px = (double*) malloc(sizeof(double) * (size_t) nb);
        double* py = (double*) malloc(sizeof(double) * (size_t) nb);
        double* pz = (double*) malloc(sizeof(double) * (size_t) nb);
        double* vx = (double*) malloc(sizeof(double) * (size_t) nb);
        double* vy = (double*) malloc(sizeof(double) * (size_t) nb);
        double* vz = (double*) malloc(sizeof(double) * (size_t) nb);
        if (px && py && pz && vx && vy && vz &&
            nbCUDABuffersDownloadBodies(st->cudaBuffers, px, py, pz, vx, vy, vz, nb) == 0)
        {
            FILE* f = fopen(path, "wb");
            if (f)
            {
                fwrite(&nb, sizeof(int), 1, f);
                for (int i = 0; i < nb; ++i) {
                    fwrite(&px[i], sizeof(double), 1, f);
                    fwrite(&py[i], sizeof(double), 1, f);
                    fwrite(&pz[i], sizeof(double), 1, f);
                }
                fclose(f);
                mw_printf("[DEBUG] dumped %d CUDA body positions to %s\n", nb, path);
            }
        }
        free(px); free(py); free(pz); free(vx); free(vy); free(vz);
        exit(0);
    }

    return NBODY_SUCCESS;
}

void nbReleaseCUDA(NBodyState* st)
{
    if (!st) return;
    if (st->cudaBuffers)
    {
        /* Pull final body state back to host before freeing. */
        (void) nbCUDAMarshalBodiesFromDevice(st);
        nbCUDAReleaseBodyBuffers(st);
    }
    st->usesCUDA = FALSE;
}

/* CPU-side LMC body integration. Replicates the math of
 * advancePosVel_LMC / advanceVelocities_LMC in nbody_plain.c so we
 * don't have to expose the static-inline helpers. */
static void nbCUDAAdvancePosVelLMC(NBodyState* st, real dt, mwvector acc, mwvector acc_i)
{
    /* x += v*dt; v += (a + a_shift) * dt/2 */
    const real dtHalf = (real) 0.5 * dt;
    mwvector dr = mw_mulvs(st->LMCvel, dt);
    mw_incaddv(st->LMCpos, dr);
    mwvector acc_total = mw_addv(acc, acc_i);
    mwvector dv = mw_mulvs(acc_total, dtHalf);
    mw_incaddv(st->LMCvel, dv);
}

static void nbCUDAAdvanceVelLMC(NBodyState* st, real dt, mwvector acc, mwvector acc_i)
{
    /* Half-kick only: v += (a + a_shift) * dt/2 */
    const real dtHalf = (real) 0.5 * dt;
    mwvector acc_total = mw_addv(acc, acc_i);
    mwvector dv = mw_mulvs(acc_total, dtHalf);
    mw_incaddv(st->LMCvel, dv);
}

/* Helper: read shiftByLMC[idx] safely, falling back to ZERO_VECTOR
 * outside the tabulated range. */
static mwvector nbCUDAShiftByLMC(const NBodyState* st, size_t idx)
{
    if (st->shiftByLMC && idx < st->nShiftLMC)
    {
        return st->shiftByLMC[idx];
    }
    mwvector zero = ZERO_VECTOR;
    return zero;
}

/* nbStepSystemCUDA: orchestrate one timestep on GPU. Mirrors the
 * CPU nbStepSystemPlain pattern (kick-drift-force-kick) with the
 * LMC body integrated on CPU and dwarf-body forces on GPU. */
NBodyStatus_int nbStepSystemCUDA(const NBodyCtx* ctx, NBodyState* st)
{
    if (!ctx || !st || !st->cudaBuffers) return NBODY_ERROR;

    const int nbody = st->nbody;
    const int nNode = nbCUDABuffersGetNNode(st->cudaBuffers);
    const real dt = ctx->timestep;
    const real barTime = (real) st->step * dt - st->previousForwardTime;

    /* shiftByLMC[step] is the LMC frame correction at the START of
     * this step; shiftByLMC[step+1] is at the END. Both are passed
     * through to the integration kernel and to the LMC body
     * integration helpers. */
    const mwvector acc_i  = nbCUDAShiftByLMC(st, st->step);
    const mwvector acc_i1 = nbCUDAShiftByLMC(st, (size_t) st->step + 1);

    /* Pack the external-potential params once per step (cheap). */
    NBodyCUDAExternalPotential pot;
    nbCUDAPackPotential(ctx, &pot);

    /* LMC parameters (only used when ctx->LMC). The LMC body's pos
     * gets updated mid-step on CPU below. */
    const int lmcActive = (ctx->LMC != FALSE);
    double lmcMass  = lmcActive ? ctx->LMCmass  : 0.0;
    double lmcScale = lmcActive ? ctx->LMCscale : 0.0;
    int lmcType = lmcActive ? NBODY_CUDA_LMC_PLUMMER : NBODY_CUDA_LMC_NONE;

    /* Per-kernel BEFORE/AFTER trace. Prints for the first 3 steps so
     * we can pinpoint exactly which kernel hangs without changing
     * timing in steady-state. */
    /* Per-kernel timing diagnostics — disabled in production (was used to
     * identify slow kernels in the useQuad path during the bring-up). */
    #define KSTART(name) ((void) 0)
    #define KEND(name)   ((void) 0)

    /* === Phase 1: first half-kick + drift on dwarf bodies (GPU).
     * Uses the OLD acc populated by the previous step's force pass
     * (or by the initial force-only pass at the end of nbInitCUDA). */
    KSTART("integration FIRST_HALF");
    if (nbCUDALaunchIntegration(st->cudaBuffers, dt, nbody,
                                NBODY_CUDA_INT_FIRST_HALF,
                                X(acc_i), Y(acc_i), Z(acc_i)) != 0)
    {
        return NBODY_ERROR;
    }
    KEND("integration FIRST_HALF");

    /* === Phase 2: move LMC body (CPU). Mirrors nbStepSystemPlain
     * lines 360-363. */
    if (lmcActive)
    {
        KSTART("LMC pos+vel CPU update");
        mwvector aExt = nbExtAcceleration(&ctx->pot, st->LMCpos, barTime);
        mwvector aDF  = dynamicalFriction_LMC(&ctx->pot, st->LMCpos, st->LMCvel,
                                              ctx->LMCmass, ctx->LMCDynaFric,
                                              barTime, ctx->coulomb_log);
        mwvector acc_LMC = mw_addv(aExt, aDF);
        nbCUDAAdvancePosVelLMC(st, dt, acc_LMC, acc_i);
        KEND("LMC pos+vel CPU update");
    }

    /* === Phase 3: force evaluation (GPU). After this, d_acc holds
     * the NEW per-body accelerations.
     * Set NBODY_CUDA_FORCE_EXACT=1 at compile time to bypass the
     * tree pipeline and use direct N² force as a debug baseline.
     * Useful for isolating tree-walk vs. external-potential bugs. */
    #ifndef NBODY_CUDA_FORCE_EXACT
      #define NBODY_CUDA_FORCE_EXACT 0
    #endif
    #if NBODY_CUDA_FORCE_EXACT
        KSTART("forceExact (DEBUG bypass tree)");
        if (nbCUDALaunchForceExact(st->cudaBuffers, nbody, ctx->eps2) != 0)
        {
            return NBODY_ERROR;
        }
        KEND("forceExact");
    #else
        KSTART("boundingBox");
        if (nbCUDALaunchBoundingBox(st->cudaBuffers, nbody, nNode) != 0)        return NBODY_ERROR;
        KEND("boundingBox");
        KSTART("buildTreeClear");
        if (nbCUDALaunchBuildTreeClear(st->cudaBuffers, nNode) != 0)            return NBODY_ERROR;
        KEND("buildTreeClear");
        KSTART("buildTree");
        if (nbCUDALaunchBuildTree(st->cudaBuffers, nbody, nNode) != 0)          return NBODY_ERROR;
        KEND("buildTree");
        KSTART("summarizationClear");
        if (nbCUDALaunchSummarizationClear(st->cudaBuffers, nbody, nNode) != 0) return NBODY_ERROR;
        KEND("summarizationClear");
        KSTART("summarization");
        if (nbCUDALaunchSummarization(st->cudaBuffers, nbody, nNode) != 0)      return NBODY_ERROR;
        KEND("summarization");
        KSTART("sort");
        if (nbCUDALaunchSort(st->cudaBuffers, nbody, nNode) != 0)               return NBODY_ERROR;
        KEND("sort");
        if (st->usesQuad)
        {
            KSTART("quadMoments");
            if (nbCUDALaunchQuadMoments(st->cudaBuffers, nNode) != 0)           return NBODY_ERROR;
            KEND("quadMoments");
            KSTART("quadPack");
            if (nbCUDALaunchQuadPack(st->cudaBuffers, nNode) != 0)              return NBODY_ERROR;
            KEND("quadPack");
        }

        KSTART("forceTree");
        if (nbCUDALaunchForceTree(st->cudaBuffers, nbody, nNode,
                                  ctx->eps2, ctx->theta,
                                  st->usesQuad ? 1 : 0,
                                  /*updateVel=*/1,
                                  dt,
                                  NBODY_CUDA_INT_FIRST_HALF) != 0)
        {
            return NBODY_ERROR;
        }
        KEND("forceTree");
    #endif

    KSTART("externalPotential");
    if (nbCUDALaunchExternalPotential(st->cudaBuffers, nbody, &pot,
                                      (NBodyCUDALMCType) lmcType,
                                      lmcMass, lmcScale, /*lmcScale2=*/0.0,
                                      X(st->LMCpos), Y(st->LMCpos), Z(st->LMCpos),
                                      /*skipLMC=*/0) != 0)
    {
        return NBODY_ERROR;
    }
    KEND("externalPotential");

    /* === Phase 4: second half-kick on dwarf bodies (GPU). Uses the
     * NEW acc just computed and the LMC frame correction at t+dt. */
    KSTART("integration SECOND_HALF");
    if (nbCUDALaunchIntegration(st->cudaBuffers, dt, nbody,
                                NBODY_CUDA_INT_SECOND_HALF,
                                X(acc_i1), Y(acc_i1), Z(acc_i1)) != 0)
    {
        return NBODY_ERROR;
    }
    KEND("integration SECOND_HALF");

    /* === Phase 5: second half-kick on LMC body (CPU). Mirrors
     * nbStepSystemPlain lines 370-373. */
    if (lmcActive)
    {
        mwvector acc_LMC = mw_addv(
            nbExtAcceleration(&ctx->pot, st->LMCpos, barTime),
            dynamicalFriction_LMC(&ctx->pot, st->LMCpos, st->LMCvel,
                                  ctx->LMCmass, ctx->LMCDynaFric,
                                  barTime, ctx->coulomb_log));
        nbCUDAAdvanceVelLMC(st, dt, acc_LMC, acc_i1);
    }

    /* Per-step body / LMC state diagnostic dump removed in production.
     * Was used during bring-up to identify when bodies started exploding.
     * Required two D→H copies (~960 KB each) per dump, which is non-trivial
     * cost even for sparse step gating. */

    st->step++;
    st->dirty = TRUE;


    return NBODY_SUCCESS;
}

NBodyStatus_int nbRunSystemCUDA(const NBodyCtx* ctx, NBodyState* st, const void* nbf_v)
{
    /* Drive the per-step loop the same way nbRunSystemPlain does:
     * nStep iterations, with periodic progress reporting and BOINC
     * checkpointing. */
    if (!ctx || !st) return NBODY_ERROR;

    /* nbf is passed as void* through nbody_cuda.h so that header
     * doesn't need to drag in the NBodyFlags definition. */
    const NBodyFlags* nbf = (const NBodyFlags*) nbf_v;

    const real Nstep = (real) ctx->nStep;

    /* Optional steps-per-second heartbeat. Enabled via env
     * NBODY_CUDA_STEP_HEARTBEAT=N (print every N steps). Useful for
     * comparing forward-sim throughput across builds. */
    int hb_every = 0;
    {
        const char* hbe = getenv("NBODY_CUDA_STEP_HEARTBEAT");
        if (hbe && hbe[0] && hbe[0] != '0') hb_every = atoi(hbe);
    }
    struct timespec hb_t0, hb_t_prev;
    if (hb_every > 0) {
        clock_gettime(CLOCK_MONOTONIC, &hb_t0);
        hb_t_prev = hb_t0;
    }

    while (st->step < ctx->nStep)
    {
        if (nbStepSystemCUDA(ctx, st) != NBODY_SUCCESS)
        {
            return NBODY_ERROR;
        }

        if (hb_every > 0 && (st->step % hb_every == 0)) {
            struct timespec t1;
            clock_gettime(CLOCK_MONOTONIC, &t1);
            double dt_total = (t1.tv_sec - hb_t0.tv_sec) + (t1.tv_nsec - hb_t0.tv_nsec) / 1e9;
            double dt_recent = (t1.tv_sec - hb_t_prev.tv_sec) + (t1.tv_nsec - hb_t_prev.tv_nsec) / 1e9;
            fprintf(stderr, "[step-hb] step=%d/%d  recent=%.2fms/step (last %d)  total=%.1fs  rate=%.1fsteps/s\n",
                    (int) st->step, (int) ctx->nStep,
                    dt_recent * 1000.0 / hb_every, hb_every,
                    dt_total, st->step / dt_total);
            hb_t_prev = t1;
        }

        /* opt #8 elaborate: in the BestLikeStart window, kick off an
         * async D2H of THIS step's bodies on a dedicated stream, and
         * only EVALUATE the likelihood for the PREVIOUS step's bodies
         * (already in the pinned buffer from last iteration's async
         * copy). The result is bestLikelihood lags by one step, which
         * doesn't change the "best across the window" computation —
         * but the per-step D2H + EMD eval now overlaps the next step's
         * GPU compute, eliminating the GPU-idle stall that the prior
         * sync-marshal-then-eval pattern had. */
        if (nbf && ctx->useBestLike)
        {
            const real frac = (real) st->step / Nstep;
            if (frac >= ctx->BestLikeStart)
            {
                /* Drain the previous step's pending async marshal, if
                 * any, into bodytab and run the eval. Skipped on the
                 * first step in the window. */
                if (nbCUDABuffersIsAsyncMarshalPending(st->cudaBuffers))
                {
                    double* sc = nbCUDAAllocSoAScratch(st->nbody);
                    if (!sc) return NBODY_ERROR;
                    int rc = nbCUDABuffersWaitAsyncBodies(
                        st->cudaBuffers,
                        sc + 0 * (size_t) st->nbody,
                        sc + 1 * (size_t) st->nbody,
                        sc + 2 * (size_t) st->nbody,
                        sc + 3 * (size_t) st->nbody,
                        sc + 4 * (size_t) st->nbody,
                        sc + 5 * (size_t) st->nbody);
                    if (rc == 0) nbCUDAUnpackSoAToBodies(sc, st->bodytab, st->nbody);
                    free(sc);
                    if (rc != 0) return NBODY_ERROR;
                    (void) nbGetLikelihoodForBest(ctx, st, nbf);
                }
                /* Kick off async D2H of THIS step's bodies — to be
                 * drained on the NEXT iteration. */
                if (nbCUDABuffersStartAsyncBodyMarshal(st->cudaBuffers) != NBODY_SUCCESS)
                {
                    return NBODY_ERROR;
                }
            }
        }


        /* Report progress to BOINC every step (cheap; mw_fraction_done
         * itself rate-limits internal updates). */
        const real frac = (real) st->step / (real) ctx->nStep;
        mw_fraction_done(frac);

        /* Checkpoint when BOINC asks OR every NBODY_CUDA_CKPT_EVERY
         * steps (whichever fires first). The step-count fallback gives
         * users live progress visibility even in standalone mode where
         * boinc_time_to_checkpoint cadence is sparse. Each checkpoint
         * is a synchronous D2H marshal (stalls the GPU pipeline) plus
         * an fwrite of the body array. Cadence raised from 50 to 500
         * — at 50 a typical 50K-step WU paid ~1000 GPU pipeline stalls;
         * 500 cuts that to ~100 while still bounding crash-recovery
         * cost to ~5 sec of work (500 × ~10 ms/step). BOINC's
         * time-based checkpoint still fires for longer-interval crash
         * recovery. */
        #ifndef NBODY_CUDA_CKPT_EVERY
          #define NBODY_CUDA_CKPT_EVERY 500
        #endif
        const int forceCheckpoint = (st->step % NBODY_CUDA_CKPT_EVERY == 0);
        if (forceCheckpoint || nbTimeToCheckpoint(ctx, st))
        {
            if (nbCUDAMarshalBodiesFromDevice(st) != NBODY_SUCCESS)
            {
                return NBODY_ERROR;
            }
            if (nbWriteCheckpoint(ctx, st))
            {
                return NBODY_ERROR;
            }
            mw_checkpoint_completed();
        }
    }

    /* Drain any final pending async-marshal from the bestLike window
     * so the last step's likelihood gets evaluated before we marshal
     * the post-loop final state. */
    if (nbf && ctx->useBestLike && nbCUDABuffersIsAsyncMarshalPending(st->cudaBuffers))
    {
        double* sc = nbCUDAAllocSoAScratch(st->nbody);
        if (!sc) return NBODY_ERROR;
        int rc = nbCUDABuffersWaitAsyncBodies(
            st->cudaBuffers,
            sc + 0 * (size_t) st->nbody,
            sc + 1 * (size_t) st->nbody,
            sc + 2 * (size_t) st->nbody,
            sc + 3 * (size_t) st->nbody,
            sc + 4 * (size_t) st->nbody,
            sc + 5 * (size_t) st->nbody);
        if (rc == 0) nbCUDAUnpackSoAToBodies(sc, st->bodytab, st->nbody);
        free(sc);
        if (rc != 0) return NBODY_ERROR;
        (void) nbGetLikelihoodForBest(ctx, st, nbf);
    }

    /* Pull final body state back to host so any post-loop CPU code
     * (likelihood/histogram/output) sees the latest pos/vel. */
    if (nbCUDAMarshalBodiesFromDevice(st) != NBODY_SUCCESS)
    {
        return NBODY_ERROR;
    }

    /* Diagnostic: cumulative max tree depth reached across the run.
     * For the Morton-buildTree path, depth > NBODY_CUDA_MAXDEPTH+1
     * means the MAXDEPTH-overflow branch fired and some bodies were
     * dropped to match legacy's atomicCAS overflow semantics. */
    {
        int maxDepth = nbCUDABuffersGetMaxDepth(st->cudaBuffers);
        fprintf(stderr, "[nbody_cuda] cumulative maxDepth=%d\n", maxDepth);
    }

    return NBODY_SUCCESS;
}

#pragma GCC diagnostic pop

#endif /* NBODY_CUDA */
