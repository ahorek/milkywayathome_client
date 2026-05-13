# CUDA backend optimization log

Notes covering the optimizations made to the CUDA nbody backend on the
`cuda-port-optimization` branch. Maintained alongside the source — when
adding a new optimization that survives review, append a row here.

The two correctness invariants for every change:

1. **Bit-identical to the legacy CUDA buildTree path** on the three
   reference WUs (`WU_1020392708`, `WU_1020960249`, `WU_1021003732`)
   running `--use-cuda --nthreads 4`.
2. **Deterministic across runs** — three back-to-back runs of the long
   WU must produce the same bit-identical `search_likelihood`.

Anything that breaks either gets reverted, no exceptions.

## Reference baselines (V100, --use-cuda --nthreads 4)

| WU              | Legacy GPU      | Final Morton + opts |  Δ      |
|-----------------|-----------------|---------------------|---------|
| WU_1020392708   | ~198 s          | 189 s               |  -9 s   |
| WU_1020960249   | ~514 s          | 483 s               | -31 s   |
| WU_1021003732   | ~519 s          | 476 s               | -43 s (-8.3%) |

All return the same bit-identical `search_likelihood` as legacy:

- WU_1020392708: `-802.605484255098872`
- WU_1020960249: `-616.882830894192125`
- WU_1021003732: `-56.612484736922859`

3-run determinism verified on the long WU after every committed
optimization.

## Pre-Morton optimizations (already in repo before the Morton work)

| Commit      | Description                                                    |
|-------------|----------------------------------------------------------------|
| `d961a49f`  | opt #1 — raise CUDA checkpoint cadence 50 → 500 steps          |
| `e55a7478`  | opt #3 + #4 — NVCC `--restrict`, PTXAS spill/lmem warnings    |
| `2418776b`  | opt #7 — drop per-kernel `cudaDeviceSynchronize`               |
| `2b1abbc5`  | multi-GPU dispatch wrapper + broader SM_ARCHS in the fatbin    |
| `3bf07cd3`  | NFW (vhalo) op order fix + `SphericalNFWerkal` halo support    |
| `ac491cc4`  | Plummer `pow_rn(tmp, 3.0)` — closed CPU↔GPU divergence on long WUs |

## Morton path foundation (this session)

The legacy `atomicCAS`-based `nbCUDABuildTreeKernel` left cell-index
allocation order dependent on warp scheduling. Pinned host memory in
the per-step pipeline (`cudaMallocHost` anywhere) perturbed CUDA
scheduling and flipped downstream FP results.

Morton replaces buildTree with a deterministic Sort → level-by-level
fused-kernel approach. **Morton is the default tree builder.** The
legacy atomicCAS path can be re-selected for debugging via the env
var `NBODY_BUILDTREE_MORTON=0`. (Default flipped because BOINC has no
clean way to set env vars, and Morton is bit-identical + deterministic
+ faster on all reference WUs.)

| Commit      | What                                                            |
|-------------|-----------------------------------------------------------------|
| `bbe8e8cc`  | Initial 21-bit Morton tree-build path (compute + sort + per-level fused kernel + counter swap). Deterministic on short/medium but mismatched legacy on long. |
| `f7f71b31`  | **Morton v5** — 128-bit Morton (42 bits/axis) + MAXDEPTH-overflow that places `d_sortedIdx[e-1]` to match legacy's lossy overflow. Bit-identical to legacy on all 3 WUs. |
| `94afdd57`  | **Three opts at once**: CUB `DeviceRadixSort` + preallocated workspace (fixed thrust merge-sort non-determinism), CUDA-graph capture of the full Morton pipeline (~50 launches → 1), opt #8 elaborate (pinned host buffer + async D2H of bodies + cudaEvent sync for bestLikelihood eval overlap). |
| `71d379c2`  | Removed dead count/scan/emit Morton kernels from initial bring-up |

### Notes on the 128-bit "Morton" type

It's a 128-bit *integer* key, not 128-bit FP. The GPU has no FP128
support and doesn't need any here. Stored as
`struct Morton128 { uint64_t lo, hi; }` — two regular 64-bit ints
side by side. All operations decompose into pairs of 64-bit integer
ops with manual carry handling for shifts. Body positions remain FP64
throughout — the Morton code is just a spatial-quantization label.

### Notes on MAXDEPTH overflow

Legacy's `nbCUDABuildTreeKernel` exits its split loop via
`depth > NBODY_CUDA_MAXDEPTH` when two bodies still share an octant at
depth 26 (0-indexed = legacy's depth 27 by its 1-indexed convention).
After loop exit, `d_child[NSUB*n+j] = i` overwrites whatever body was
at that slot — net effect: one body retained, the other lost,
errorCode = 1 set.

Morton v5 replicates this exact behavior. At MAXDEPTH overflow in the
fused kernel:
```c
d_child[slot] = d_sortedIdx[e - 1];
d_treeStatus->errorCode = 1;
```
The heuristic "keep the highest-body_idx body in the octant range"
(matching legacy's stride-based thread arrival order) produced a
bit-identical match on the first try.

## forceTree memory-bound optimizations

Profile (V100, short WU, nsys): forceTree is **92.2 % of GPU time**
(117 s of 127 s GPU). Everything else combined is <8 %. All
optimization effort below targets forceTree's per-cell-visit memory
traffic and shared-memory layout.

| Commit      | Δ on long WU  | Description                                  |
|-------------|---------------|----------------------------------------------|
| `074f04d6`  | -1 s          | `__ldg()` for per-body initial pos read (stable within a forceTree launch; mutated only by Integration, which runs *after*). `__shfl_sync` register-broadcast replacing the per-warp shared-memory leader broadcast (4 stores + 128 reads + `__syncwarp` per cell visit → 5 shfls). |
| `b692c035`  | -5 s          | Pack the 6 separate `d_quad**` arrays into `d_quadPacked` (8 doubles/cell, slots 6..7 padded) so the 6 accept-case quad reads hit 1 cache line instead of 6. New post-QuadMoments pack kernel. |
| `d852f009`  | **-19 s**     | **Removed dead `__shared__` per-warp quad stacks + `rootQXX..ZZ` snapshots.** They were written on every cell-open path and at kernel start, but read nowhere — the accept path uses the visited node's own quad directly. Freed 9.7 KB shared mem; removed 6 volatile global loads per cell-open path. Biggest single perf win of the session. |
| `32927419`  | -3 s          | Pack `(posX, posY, posZ, mass)` per index into `d_posMassPacked` (4 doubles = 32 bytes / index). Leader read becomes 1 cache line instead of 4 separate loads from `d_pos*`/`d_masses`. |
| `48508625`  | -3 s          | **Mega-pack** — combine pos+mass + critRadii + quad into a single 128-byte / cell layout (`d_cellPacked`, 16 doubles / index). All ForceTree cell-data reads (leader's pos+mass, all-lane broadcast of critRadii, all-lane broadcast of 6 quad components on accept) hit the **same cache line** for cell n. |
| `3a40bd74`  | 0             | Cleanup — removed `d_posMassPacked` + `d_quadPacked` + their kernels/launchers/header decls + their old params from the forceTree signature. All superseded by `d_cellPacked`. -159 lines. |

### Final mega-pack layout per cell

```
d_cellPacked[n * 16 + 0..2]  = posX, posY, posZ
d_cellPacked[n * 16 + 3]     = mass
d_cellPacked[n * 16 + 4]     = critRadii (rc² after summarization; 0 for bodies)
d_cellPacked[n * 16 + 5..10] = quad XX, XY, XZ, YY, YZ, ZZ
d_cellPacked[n * 16 + 11..15] = padding (0)
```

Refreshed once per step by `nbCUDACellPackKernel` after `QuadMoments`.
Body slots zero out critRadii and quad so the same access path serves
both bodies and cells without a branch.

## Tried-and-reverted (documented for future readers)

| Idea | Why reverted |
|------|--------------|
| `__launch_bounds__(NBODY_CUDA_BLOCK, 4)` on forceTree | Drops regs 76→64 on V100 (50% occupancy), but **breaks bit-identity on long WU** (Δ 0.04). PTX op counts identical to baseline — the divergence is from ptxas SASS-level rescheduling of independent FP ops under different register pressure. Not fixable in source without hand-written PTX. |
| `__launch_bounds__(NBODY_CUDA_BLOCK, 3)` (same nominal occupancy as natural) | Bit-identical AND deterministic but consistently **8 s slower** on the long WU (516 s vs 508 s). Fewer regs forced more register-reuse copies despite no `LOCAL:0` spill. |
| `__ldg()` on per-cell `d_pos`/`d_masses` reads in forceTree | Breaks bit-identity (Δ 0.02 on long WU). Cell range of `d_pos`/`d_masses` IS mutated within a step by Summarization; the read-only cache can hold stale values across the cell-recycle boundary. |
| Removing the NaN-check on packed quad reads | Risky — kept the defensive check. |

## What's left (not pursued)

| Idea | Expected win | Why deferred |
|------|--------------|--------------|
| Top-of-tree caching in shared mem (root + 8 level-1 children) | ~1-2 s | Profile says only ~70 GB of cumulative bandwidth saved; high code complexity for the win. |
| Whole-step CUDA graph capture (everything, not just Morton) | <2 s | Per-step argument changes (LMC pos, branch flag, dt) make capture re-instantiation per step prohibitive; long-kernel launch overhead is already amortized. |
| Raise `NBODY_CUDA_MAXDEPTH` for higher BH precision | precision, not perf | Diverges from legacy by retaining the bodies legacy drops; invalidates the bit-identical invariant. Would also need lifting in forceTree's safety check and per-warp depth-stack size. |
| Algorithmic change (FMM, kd-tree, …) | unknown | Major rewrite; not a drop-in. |

## Profiling

Measured 2026-05-13, V100 SXM2 16GB, short WU = WU_1020392708.

```
nsys profile --trace=cuda,nvtx ./milkyway_nbody ...
nsys stats --report cuda_gpu_kern_sum ...
```

| Kernel                      | GPU time | % of GPU |
|-----------------------------|---------:|---------:|
| `nbCUDAForceTreeKernel`     |  117.0 s |  92.2 %  |
| `nbCUDAQuadMomentsKernel`   |    4.7 s |   3.7 %  |
| `nbCUDASummarizationKernel` |    3.0 s |   2.4 %  |
| `nbCUDACellPackKernel`      |   0.95 s |   0.7 %  |
| `nbCUDAExternalPotentialKernel` |  0.62 s | 0.5 %  |
| `nbCUDABoundingBoxKernel`   |   0.30 s |   0.2 %  |
| `nbCUDAIntegrationKernel`   |   0.12 s |   0.1 %  |
| (everything else combined)  |  <0.1 s  |  ~0 %    |

forceTree per-launch: **6.88 ms** × 17,008 launches.

ForceTree resource usage on sm_70 after the cleanups:
- 67-73 registers / thread (down from 76 baseline)
- 3,340 bytes shared / block (down from 13,372 baseline)
- 0 bytes local memory spill

## Build & test recipe

```bash
# Build (in build/ created by build_cuda.sh)
cd build && make -j8 milkyway_nbody

# Run (Morton tree is now the default)
./bin/milkyway_nbody \
    -f nbody_parameters.lua -h histogram.txt \
    --seed <seed> -np 12 -p <12 params> \
    --nthreads 4 --use-cuda

# Force legacy buildTree (debug only)
NBODY_BUILDTREE_MORTON=0 ./bin/milkyway_nbody ...

# 3-run determinism check
bash /tmp/morton_3run.sh WU_1021003732 milkyway_nbody.morton

# Profile (drops a .nsys-rep)
nsys profile --trace=cuda,nvtx -o /tmp/nsys_out ./milkyway_nbody ...
nsys stats --report cuda_gpu_kern_sum /tmp/nsys_out.nsys-rep
```

Diagnostic env vars:
- `NBODY_BUILDTREE_MORTON=0` — force legacy buildTree (Morton is default)
- `NBODY_CUDA_STEP_HEARTBEAT=N` — print steps-per-second every N steps
- `NBODY_BUILDTREE_MORTON_PROFILE=1` — per-phase timing for the first 5 Morton builds
