/*
 * CUDA backend public API for the milkyway nbody application.
 *
 * Mirrors the structure of nbody_cl.h. The runtime selects the CUDA
 * path when st->usesCUDA is true; otherwise the CPU path runs.
 *
 * This header is deliberately self-contained — it forward-declares the
 * types it needs instead of pulling in nbody_types.h or milkyway_math.h.
 * That keeps the .cu translation unit's preprocessor state small enough
 * for nvcc to compile cleanly (the milkyway/openpa header chain
 * otherwise drags in inline asm and GCC builtins that nvcc can choke
 * on). Callers in plain C see the same forward declarations and use
 * them through pointers, which is sufficient because nvcc and the C
 * compiler agree on struct pointer ABI.
 */

#ifndef _NBODY_CUDA_H_
#define _NBODY_CUDA_H_

#include "nbody_config.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Forward declarations only. The full struct definitions live in
 * nbody_types.h and are only needed by C translation units that
 * actually touch fields. */
struct NBodyCtx_s;
struct NBodyState_s;
typedef struct NBodyCtx_s   NBodyCtx;
typedef struct NBodyState_s NBodyState;

/* NBodyStatus is a bit-flag enum in nbody_types.h. Across the C/C++
 * boundary an int is layout-compatible. We don't enumerate the names
 * here to avoid pulling in the enum definition. */
typedef int NBodyStatus_int;

#if NBODY_CUDA

/* Opaque CUDA-side state. Full definitions are private to nbody_cuda.cu. */
struct NBodyCUDAContext;
struct NBodyCUDAKernels;
struct NBodyCUDABuffers;

/* Probe for a usable CUDA device and populate state. Returns 0 (success)
 * or a non-zero NBodyStatus bit-flag. Must be called once during setup,
 * before nbStepSystem(). */
NBodyStatus_int nbInitCUDA(const NBodyCtx* ctx, NBodyState* st);

/* Tear down CUDA state and free device buffers. Safe to call on a state
 * that was never initialized (no-op). */
void nbReleaseCUDA(NBodyState* st);

/* Per-step entry point. Equivalent to nbStepSystemPlain / nbStepSystemCL
 * but runs the force + integration on the CUDA device. */
NBodyStatus_int nbStepSystemCUDA(const NBodyCtx* ctx, NBodyState* st);

/* Run the full simulation loop on CUDA (handles checkpointing,
 * progress reporting, etc.). Equivalent to nbRunSystemCL.
 *
 * `nbf` is the flags struct so we can call nbGetLikelihoodForBest
 * per step in the BestLikeStart window — the same code CPU runs in
 * its main loop. Without that, useBestLike-controlled output ends
 * up reporting the worst-case sentinel because st->bestLikelihood
 * never gets updated. */
NBodyStatus_int nbRunSystemCUDA(const NBodyCtx* ctx, NBodyState* st, const void* nbf);

#endif /* NBODY_CUDA */

#ifdef __cplusplus
}
#endif

#endif /* _NBODY_CUDA_H_ */
