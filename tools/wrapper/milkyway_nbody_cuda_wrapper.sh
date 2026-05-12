#!/usr/bin/env bash
# milkyway_nbody_cuda_wrapper.sh
#
# Multi-GPU dispatch wrapper for the CUDA-built milkyway_nbody binary.
# Designed for BOINC clients that run the nbody app via app_info.xml as
# a "CPU" application (because the project scheduler does not currently
# dispatch nbody work as GPU work).
#
# What it does:
#   1. Detects available CUDA GPUs (nvidia-smi, or honors a pre-set
#      CUDA_VISIBLE_DEVICES if BOINC/the user already set one).
#   2. Picks the first free (gpu_index, instance) slot using flock.
#      Slots are per-uid lock files; the kernel auto-releases the lock
#      when the exec'd binary exits, so crashes/aborts never strand a
#      slot.
#   3. Exports CUDA_VISIBLE_DEVICES so the app sees only that one GPU
#      (as device 0) and exec's the binary with the original BOINC
#      arguments plus --use-cuda.
#
# Tunables (env vars; the wrapper sources a sibling
# milkyway_nbody_cuda_wrapper.conf if present, or honors values from
# the parent process environment — BOINC's app_config.xml schema does
# not support per-app env vars, so don't try to set them there):
#   MW_TASKS_PER_GPU   per-GPU concurrency  (default: 1)
#   MW_GPU_INDICES     comma-list allowlist (default: all detected GPUs)
#                      e.g. "0,2,3" to skip the display GPU at index 1
#   MW_FALLBACK_CPU    if 1, drop --use-cuda when no GPU detected
#                      (default: 0 — error out)
#   MW_NBODY_BIN       path to milkyway_nbody binary
#                      (default: ./milkyway_nbody.bh, then sibling of
#                       this script)
#   MW_LOCK_DIR        override lock-file directory (advanced)
#
# Operational note: set BOINC's <project_max_concurrent> equal to
#   N_GPUs × MW_TASKS_PER_GPU
# and the wrapper will never need to queue. If oversubscribed, it
# falls back to a blocking flock so tasks correctly serialize, but
# BOINC will count the queueing tasks as "running."

set -euo pipefail

log() { printf '[mw_cuda_wrapper] %s\n' "$*" >&2; }
die() { log "ERROR: $*"; exit 1; }

SCRIPT_PATH=$(readlink -f -- "${BASH_SOURCE[0]}" 2>/dev/null || echo "${BASH_SOURCE[0]}")
SCRIPT_DIR=$(dirname -- "$SCRIPT_PATH")

# Source optional config from the first location that exists. Search
# order (most specific first):
#   1. $MW_WRAPPER_CONF                                — explicit override
#   2. $SCRIPT_DIR/milkyway_nbody_cuda_wrapper.conf    — sibling of script
#                                                        (works if listed in
#                                                         app_info.xml so BOINC
#                                                         hard-links it per-slot)
#   3. ${XDG_CONFIG_HOME:-$HOME/.config}/milkyway_nbody_cuda_wrapper.conf
#                                                      — per-user (no app_info
#                                                        change needed)
#   4. /etc/milkyway_nbody_cuda_wrapper.conf            — system-wide
for _candidate in \
    "${MW_WRAPPER_CONF:-}" \
    "$SCRIPT_DIR/milkyway_nbody_cuda_wrapper.conf" \
    "${XDG_CONFIG_HOME:-$HOME/.config}/milkyway_nbody_cuda_wrapper.conf" \
    "/etc/milkyway_nbody_cuda_wrapper.conf"
do
    if [[ -n "$_candidate" && -r "$_candidate" ]]; then
        # shellcheck disable=SC1090
        . "$_candidate"
        log "loaded config from $_candidate"
        break
    fi
done
unset _candidate

# ------------------------- defaults --------------------------
MW_TASKS_PER_GPU="${MW_TASKS_PER_GPU:-1}"
MW_FALLBACK_CPU="${MW_FALLBACK_CPU:-0}"

# ------------------------- locate the binary -----------------
if [[ -n "${MW_NBODY_BIN:-}" ]]; then
    BIN="$MW_NBODY_BIN"
elif [[ -x ./milkyway_nbody.bh ]]; then
    BIN=./milkyway_nbody.bh
elif [[ -x "$SCRIPT_DIR/milkyway_nbody.bh" ]]; then
    BIN="$SCRIPT_DIR/milkyway_nbody.bh"
elif [[ -x "$SCRIPT_DIR/milkyway_nbody" ]]; then
    BIN="$SCRIPT_DIR/milkyway_nbody"
else
    die "milkyway_nbody binary not found (cwd, \$MW_NBODY_BIN, or $SCRIPT_DIR)"
fi
[[ -x "$BIN" ]] || die "$BIN is not executable"

# ------------------------- detect GPUs -----------------------
declare -a GPUS=()
if [[ -n "${MW_GPU_INDICES:-}" ]]; then
    IFS=',' read -ra GPUS <<<"$MW_GPU_INDICES"
elif [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    # Respect what BOINC/parent already set (whitespace/comma-tolerant).
    IFS=', ' read -ra GPUS <<<"$CUDA_VISIBLE_DEVICES"
elif command -v nvidia-smi >/dev/null 2>&1; then
    mapfile -t GPUS < <(nvidia-smi --query-gpu=index \
                                    --format=csv,noheader,nounits 2>/dev/null)
fi
# Strip whitespace from each entry.
for ((idx=0; idx<${#GPUS[@]}; idx++)); do
    GPUS[idx]="${GPUS[idx]// /}"
done
N_GPUS=${#GPUS[@]}

if (( N_GPUS == 0 )); then
    if [[ "$MW_FALLBACK_CPU" == "1" ]]; then
        log "no CUDA GPUs visible — running on CPU (MW_FALLBACK_CPU=1)"
        # Strip any existing --use-cuda from args so we don't error in
        # the binary when no CUDA device is present.
        ARGS=()
        for a in "$@"; do [[ "$a" == "--use-cuda" ]] || ARGS+=("$a"); done
        exec "$BIN" "${ARGS[@]}"
    fi
    die "no CUDA GPUs detected (set MW_FALLBACK_CPU=1 to run on CPU instead)"
fi

# ------------------------- lock-dir resolution ---------------
if [[ -n "${MW_LOCK_DIR:-}" ]]; then
    LOCK_DIR="$MW_LOCK_DIR"
elif [[ -n "${XDG_RUNTIME_DIR:-}" && -d "$XDG_RUNTIME_DIR" && -w "$XDG_RUNTIME_DIR" ]]; then
    LOCK_DIR="$XDG_RUNTIME_DIR/milkyway_gpu"
else
    LOCK_DIR="${TMPDIR:-/tmp}/milkyway_gpu_$(id -u)"
fi
mkdir -p "$LOCK_DIR"
chmod 700 "$LOCK_DIR" 2>/dev/null || true

# ------------------------- slot acquisition ------------------
# Try every (gpu, instance) slot non-blocking. First free wins. If all
# are busy, fall back to a blocking lock on slot 0 so the task waits
# correctly instead of failing — BOINC will count it as running.
acquired_gpu=""
for gpu in "${GPUS[@]}"; do
    for ((i=0; i<MW_TASKS_PER_GPU; i++)); do
        lockfile="$LOCK_DIR/slot_g${gpu}_i${i}.lock"
        exec {lockfd}>"$lockfile"
        if flock -n "$lockfd"; then
            acquired_gpu="$gpu"
            acquired_inst="$i"
            break 2
        fi
        # Couldn't lock; release this fd before trying the next slot.
        eval "exec ${lockfd}>&-"
    done
done

if [[ -z "$acquired_gpu" ]]; then
    # All slots occupied — block on the first slot. The wrapper will
    # resume here when one frees, then re-export CUDA_VISIBLE_DEVICES
    # for that slot's GPU and exec the binary.
    log "all $((N_GPUS * MW_TASKS_PER_GPU)) slots busy — blocking on slot g${GPUS[0]}_i0"
    lockfile="$LOCK_DIR/slot_g${GPUS[0]}_i0.lock"
    exec {lockfd}>"$lockfile"
    flock "$lockfd"
    acquired_gpu="${GPUS[0]}"
    acquired_inst=0
fi

log "acquired GPU=$acquired_gpu inst=$acquired_inst (of N_GPUS=$N_GPUS, tasks/gpu=$MW_TASKS_PER_GPU)"

# ------------------------- launch ----------------------------
# CUDA_VISIBLE_DEVICES masks all other GPUs; the app sees the chosen
# one as device 0. We exec rather than spawn so BOINC keeps monitoring
# the same PID, and so the kernel auto-releases the flock-held fd when
# the exec'd binary exits (no trap handler / stale-lock GC needed).
export CUDA_VISIBLE_DEVICES="$acquired_gpu"

# Add --use-cuda only if not already in the BOINC-passed args.
have_use_cuda=0
for a in "$@"; do [[ "$a" == "--use-cuda" ]] && have_use_cuda=1 && break; done
if (( have_use_cuda )); then
    exec "$BIN" "$@"
else
    exec "$BIN" "$@" --use-cuda
fi
