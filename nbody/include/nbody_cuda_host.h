/*
 * Plain-C host-side helpers for the CUDA backend.
 *
 * Unlike nbody_cuda.h / nbody_cuda_buffers.h, this header is meant to
 * be included only by C translation units (never by nbody_cuda.cu).
 * It pulls in nbody_types.h so callers get the full NBodyState type.
 * The implementations live in nbody_cuda_host.c and call into the
 * .cu side through the POD-only nbody_cuda_buffers.h API.
 */

#ifndef _NBODY_CUDA_HOST_H_
#define _NBODY_CUDA_HOST_H_

#include "nbody_config.h"

#if NBODY_CUDA

#include "nbody_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Allocate device buffers (if not already) and upload the current
 * st->bodytab into them. After success, st->cudaBuffers holds the
 * device-side handle.
 *
 *   nNode - tree-node capacity. Pass 0 for exact-mode runs (pos/mass
 *           sized to nbody only). Pass the result of
 *           nbCUDABuffersComputeNNode for tree-code runs (pos/mass
 *           sized to nNode+1 so cells have storage). */
NBodyStatus nbCUDAMarshalBodiesToDevice(NBodyState* st, int nNode);

/* Download pos/vel from the device buffers back into st->bodytab. */
NBodyStatus nbCUDAMarshalBodiesFromDevice(NBodyState* st);

/* Release the device buffers and clear st->cudaBuffers. Safe to call
 * if buffers were never allocated. */
void nbCUDAReleaseBodyBuffers(NBodyState* st);

#ifdef __cplusplus
}
#endif

#endif /* NBODY_CUDA */

#endif /* _NBODY_CUDA_HOST_H_ */
