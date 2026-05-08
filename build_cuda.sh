#!/usr/bin/env bash
# Build helper for the CUDA-enabled milkyway_nbody app.
# Configures cmake with BOINC, CUDA, and target SM architectures
# and runs the build.
#
# Usage:
#   ./build_cuda.sh [-b BOINC_ROOT] [-c CUDA_TOOLKIT] [-a SM_ARCHS]
#                   [-d BUILD_DIR] [-j N] [-h]
#
# Examples:
#   ./build_cuda.sh                                 # all defaults
#   ./build_cuda.sh -b /opt/boinc -c /usr/local/cuda
#   ./build_cuda.sh -a "70;80;90"                   # V100, A100, H100
#   ./build_cuda.sh -d build_v100 -j 16

set -euo pipefail

# ------------------------- defaults (override via flags) ---------------------
BOINC_ROOT="${BOINC_ROOT:-/home/ian/builds/boinc}"
CUDA_TOOLKIT="${CUDA_TOOLKIT:-/usr/local/cuda-12.9}"
SM_ARCHS="${SM_ARCHS:-70;80;87}"
BUILD_DIR="${BUILD_DIR:-build}"
JOBS="${JOBS:-$(nproc 2>/dev/null || echo 4)}"
# -----------------------------------------------------------------------------

usage() {
    cat <<EOF
Usage: $0 [options]

Options:
  -b BOINC_ROOT      Path to BOINC source tree (default: $BOINC_ROOT)
                     If you installed via 'apt install libboinc-dev' or
                     'make install' to /usr/local, omit this flag — the
                     cmake search will fall back to standard system paths.
  -c CUDA_TOOLKIT    CUDA toolkit root (parent of bin/nvcc and lib64)
                     (default: $CUDA_TOOLKIT)
  -a SM_ARCHS        Semicolon-separated list of SM compute capabilities
                     to embed into the binary (default: "$SM_ARCHS")
  -d BUILD_DIR       Build directory (default: $BUILD_DIR)
  -j N               Parallel make jobs (default: $JOBS)
  -h                 Show this help

Common SM archs:
  70  V100, Titan V          (Volta)
  72  Xavier
  75  T4, RTX 20xx           (Turing)
  80  A100, A30              (Ampere data-center)
  86  RTX 30xx, A40          (Ampere consumer)
  87  Jetson Orin
  89  RTX 40xx, L40          (Ada)
  90  H100, H200             (Hopper)

Examples:
  $0 -b /usr/local                      # system-installed BOINC
  $0 -c /opt/cuda                       # custom CUDA path
  $0 -a "70;80"                         # V100 + A100 only
  $0 -d build_a100 -a 80 -j 32          # named build dir, 32 jobs
EOF
    exit 0
}

while getopts ":b:c:a:d:j:h" opt; do
    case $opt in
        b) BOINC_ROOT="$OPTARG" ;;
        c) CUDA_TOOLKIT="$OPTARG" ;;
        a) SM_ARCHS="$OPTARG" ;;
        d) BUILD_DIR="$OPTARG" ;;
        j) JOBS="$OPTARG" ;;
        h) usage ;;
        \?) echo "Unknown option: -$OPTARG" >&2; usage ;;
        :)  echo "Option -$OPTARG requires an argument" >&2; usage ;;
    esac
done

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
SOURCE_DIR="$SCRIPT_DIR"
NVCC_PATH="$CUDA_TOOLKIT/bin/nvcc"

# ------------------------------ sanity checks --------------------------------
if [[ ! -f "$SOURCE_DIR/CMakeLists.txt" ]]; then
    echo "ERROR: $SOURCE_DIR/CMakeLists.txt not found." >&2
    echo "       This script must live in the source root." >&2
    exit 1
fi

if [[ ! -x "$NVCC_PATH" ]]; then
    echo "ERROR: nvcc not found at $NVCC_PATH" >&2
    echo "       Pass -c /path/to/cuda-toolkit (the dir containing bin/nvcc)" >&2
    exit 1
fi

if [[ ! -d "$BOINC_ROOT" ]]; then
    echo "WARN:  -b BOINC_ROOT=$BOINC_ROOT does not exist." >&2
    echo "       If BOINC is system-installed (apt install libboinc-dev)," >&2
    echo "       this is OK — cmake will fall back to /usr/include/boinc." >&2
fi
# -----------------------------------------------------------------------------

mkdir -p "$BUILD_DIR"

cat <<EOF
===== Build configuration =====
  Source:        $SOURCE_DIR
  Build dir:     $BUILD_DIR
  BOINC root:    $BOINC_ROOT
  CUDA toolkit:  $CUDA_TOOLKIT
  SM archs:      $SM_ARCHS
  Parallel jobs: $JOBS
===============================

EOF

cd "$BUILD_DIR"

cmake "$SOURCE_DIR" \
    -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
    -DCMAKE_BUILD_TYPE=Release \
    -DBOINC_APPLICATION=ON \
    -DBOINC_ROOT="$BOINC_ROOT" \
    -DSEPARATION=OFF \
    -DNBODY=ON \
    -DNBODY_GL=OFF \
    -DNBODY_OPENMP=ON \
    -DNBODY_OPENCL=OFF \
    -DNBODY_CRLIBM=ON \
    -DNBODY_CUDA=ON \
    -DNBODY_STATIC=OFF \
    -DCMAKE_CUDA_COMPILER="$NVCC_PATH" \
    -DCUDA_ARCHITECTURES="$SM_ARCHS"

echo
echo "===== Configure done — building ====="
echo

cmake --build . -j "$JOBS" --target milkyway_nbody

BIN_PATH="$(realpath "$BUILD_DIR/bin/milkyway_nbody" 2>/dev/null || echo "$PWD/bin/milkyway_nbody")"
cat <<EOF

===== Build complete =====
Binary: $BIN_PATH
EOF
ls -la "$BIN_PATH"
