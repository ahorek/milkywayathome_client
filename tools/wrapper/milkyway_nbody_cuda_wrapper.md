# milkyway_nbody_cuda_wrapper.sh — usage

A multi-GPU dispatch wrapper for the CUDA-built `milkyway_nbody`
binary. Designed for BOINC clients that run the nbody app via
`app_info.xml` as a "CPU" application (the project scheduler does not
currently dispatch nbody work as GPU work).

The wrapper assigns each task to a specific CUDA GPU using a
file-locking slot system, so multiple concurrent BOINC tasks can share
one or more GPUs without colliding.

---

## Files

Both live in the project source root:

```
milkyway_nbody_cuda_wrapper.sh        the wrapper script
milkyway_nbody_cuda_wrapper.conf      optional sibling config (sourced
                                       at startup if present)
```

---

## What it does on each invocation

1. Locates the `milkyway_nbody` binary (default search: `./milkyway_nbody.bh`,
   then a sibling of the wrapper).
2. Detects available CUDA GPUs (`nvidia-smi`, or honors a pre-set
   `CUDA_VISIBLE_DEVICES`, or an `MW_GPU_INDICES` allowlist).
3. Picks the first free `(gpu_index, instance)` slot using a non-blocking
   `flock` on a per-slot lock file. If every slot is busy, falls back
   to a blocking `flock` so the task waits in queue rather than
   failing.
4. Exports `CUDA_VISIBLE_DEVICES=<chosen_gpu>` (the app sees that one
   GPU as device 0).
5. `exec`s the binary with the original arguments plus `--use-cuda`
   (added only if not already present).

The Linux kernel auto-releases the flock when the exec'd binary exits
or is killed — no trap handlers, no stale-lock cleanup needed.

---

## Configuration (env vars)

| Variable           | Default                          | Purpose |
|--------------------|----------------------------------|---------|
| `MW_TASKS_PER_GPU` | `1`                              | Per-GPU concurrency. Set to 2+ to allow multiple tasks to share each GPU. |
| `MW_GPU_INDICES`   | *all detected*                   | Comma-separated allowlist, e.g. `"0,2,3"` to skip GPU 1 (display GPU). |
| `MW_FALLBACK_CPU`  | `0`                              | If `1`, drop `--use-cuda` and run on CPU when no GPU is detected (otherwise: error). |
| `MW_NBODY_BIN`     | `./milkyway_nbody.bh` then sibling | Path to the milkyway_nbody binary. |
| `MW_LOCK_DIR`      | auto (see below)                 | Override the lock-file directory. |

**Lock dir auto-resolution** (per-uid, so `ian` and `boinc` users don't collide):
1. `$MW_LOCK_DIR` if set
2. `$XDG_RUNTIME_DIR/milkyway_gpu` (typical on systemd desktops)
3. `${TMPDIR:-/tmp}/milkyway_gpu_<uid>`

You can set any of these in:

- A `milkyway_nbody_cuda_wrapper.conf` file — see the search order in the BOINC integration section below
- The BOINC client's environment (e.g. systemd `Environment=` line or shell startup) — applies to ALL BOINC tasks on the host
- Your shell environment (for manual standalone testing)

---

## BOINC integration

### app_info.xml

Treat the wrapper as the executable. Both files (wrapper + binary)
need to be available in the slot directory.

```xml
<app_info>
  <app>
    <name>milkyway_nbody</name>
  </app>

  <file>
    <name>milkyway_nbody_cuda_wrapper.sh</name>
    <executable/>
  </file>
  <file>
    <name>milkyway_nbody.bh</name>
    <executable/>
  </file>

  <app_version>
    <app_name>milkyway_nbody</app_name>
    <version_num>195</version_num>
    <api_version>7.7.0</api_version>
    <file_ref>
      <file_name>milkyway_nbody_cuda_wrapper.sh</file_name>
      <main_program/>
    </file_ref>
    <file_ref>
      <file_name>milkyway_nbody.bh</file_name>
    </file_ref>
  </app_version>
</app_info>
```

### app_config.xml

Set `<project_max_concurrent>` to **N_GPUs × MW_TASKS_PER_GPU** so
BOINC never launches more tasks than there are slots. The wrapper's
blocking-flock fallback will keep things correct if you exceed this
by mistake, but BOINC will then count queueing tasks as "running" and
the dashboard will be misleading.

```xml
<app_config>
  <app>
    <name>milkyway_nbody</name>
    <max_concurrent>4</max_concurrent>     <!-- e.g. 2 GPUs × 2 tasks -->
  </app>
  <project_max_concurrent>4</project_max_concurrent>
</app_config>
```

**Tunables go in a `.conf` file**, not in `app_config.xml` (BOINC's
`app_config.xml` schema doesn't support per-app env vars). The wrapper
searches these locations in order and sources the first one it finds:

1. `$MW_WRAPPER_CONF` — explicit path override
2. `<script-dir>/milkyway_nbody_cuda_wrapper.conf` — sibling of the wrapper
3. `${XDG_CONFIG_HOME:-$HOME/.config}/milkyway_nbody_cuda_wrapper.conf` — per-user
4. `/etc/milkyway_nbody_cuda_wrapper.conf` — system-wide

`milkyway_nbody_cuda_wrapper.conf` example:

```sh
MW_TASKS_PER_GPU=2
MW_GPU_INDICES="0,2"
```

**Two practical deployment styles:**

- **Per-user (recommended for hosts you own):** drop the `.conf` in
  `~/.config/`. No `app_info.xml` change needed; works for all WUs.

- **Per-project (good for shipping the same config to many hosts):**
  drop the `.conf` next to the wrapper in the project dir AND list it
  in `app_info.xml` so BOINC hard-links it into each slot:

  ```xml
  <file>
    <name>milkyway_nbody_cuda_wrapper.conf</name>
  </file>
  ...
  <app_version>
    ...
    <file_ref>
      <file_name>milkyway_nbody_cuda_wrapper.conf</file_name>
    </file_ref>
  </app_version>
  ```

  (without the `<file_ref>`, BOINC doesn't link it into slots, so the
  wrapper's `<script-dir>` lookup wouldn't see it).

To set the tunables globally for all BOINC tasks without a `.conf`,
export them in the environment that starts the BOINC client (e.g. the
systemd unit's `Environment=` line or your shell startup).

---

## Standalone usage (testing without BOINC)

```sh
cd <project-source>
./milkyway_nbody_cuda_wrapper.sh \
    -f nbody_parameters.lua -h histogram.txt \
    --seed 230338636 -np 12 -p <params...> \
    --nthreads 4
```

The wrapper appends `--use-cuda` itself; you don't need to pass it.

To override per-invocation:

```sh
MW_TASKS_PER_GPU=2 MW_GPU_INDICES=0,2 ./milkyway_nbody_cuda_wrapper.sh ...
```

The wrapper logs its slot acquisition to stderr (visible in BOINC's
`stderr.txt`):

```
[mw_cuda_wrapper] acquired GPU=0 inst=0 (of N_GPUS=2, tasks/gpu=2)
```

Or, when the queueing fallback kicks in:

```
[mw_cuda_wrapper] all 4 slots busy — blocking on slot g0_i0
[mw_cuda_wrapper] acquired GPU=0 inst=0 (of N_GPUS=2, tasks/gpu=2)
```

---

## How the slot system works

For `N_GPUs=2, MW_TASKS_PER_GPU=2` the wrapper manages 4 slot files:

```
$LOCK_DIR/slot_g0_i0.lock
$LOCK_DIR/slot_g0_i1.lock
$LOCK_DIR/slot_g1_i0.lock
$LOCK_DIR/slot_g1_i1.lock
```

Each invocation tries `flock -n` on each in order. The first slot it
locks wins; the wrapper exports `CUDA_VISIBLE_DEVICES=<that gpu>` and
exec's the binary. The locked file descriptor is inherited across the
exec; the kernel releases it automatically when the binary exits or
is killed (including SIGKILL on BOINC abort, system shutdown, etc.).

This means:

- No PID files, no manual cleanup, no stale-lock GC.
- `kill -9` on the binary releases the slot immediately.
- A reboot starts with a clean slate (lock files persist on disk but
  hold no advisory lock until reopened).

---

## Sanity checks before deploying

1. `bash -n milkyway_nbody_cuda_wrapper.sh` — syntax check.
2. `MW_NBODY_BIN=/bin/echo ./milkyway_nbody_cuda_wrapper.sh test` —
   prove GPU detection + slot acquisition without launching anything heavy.
3. With the real binary: confirm the wrapper's `acquired GPU=N` line
   appears at the top of BOINC's `stderr.txt` for every task and that
   the `<search_likelihood>` value matches a baseline run.

---

## Limitations

- Linux/macOS only (`flock`-based). Windows BOINC clients would need a
  different locking primitive.
- BOINC remains unaware of the GPU dispatching. Any project-level
  per-GPU bookkeeping (e.g. coproc reservations) is handled
  exclusively by `<project_max_concurrent>` and the wrapper's slot
  count. They must agree.
- If you change `MW_TASKS_PER_GPU` while tasks are running, BOINC's
  `<project_max_concurrent>` is no longer in sync until you adjust it
  too.
