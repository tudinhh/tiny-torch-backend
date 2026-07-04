#!/bin/bash
set -e  # Exit immediately on any error
set -u  # Treat unset variables as errors
set -o pipefail  # Catch failures in piped commands

# ==============================================================================
# compile.sh — AOT compilation pipeline: PyTorch model -> standalone .so
# ==============================================================================

MODEL_NAME="model"
BUILD_DIR="build"

MLIR_LINALG="${BUILD_DIR}/${MODEL_NAME}.linalg.mlir"
MLIR_LLVM="${BUILD_DIR}/${MODEL_NAME}.llvm.mlir"
LLVM_IR="${BUILD_DIR}/${MODEL_NAME}.ll"
OUTPUT_SO="${BUILD_DIR}/${MODEL_NAME}.so"

mkdir -p "${BUILD_DIR}"

echo "=== [1/4] Generating Linalg MLIR from PyTorch model ==="
python3 model_to_mlir.py --output "${MLIR_LINALG}"

echo "=== [2/4] Lowering Linalg -> tiny_rt -> LLVM dialect via tiny-opt ==="
tiny-opt "${MLIR_LINALG}" \
    --convert-linalg-to-tiny-rt \
    --lower-tiny-rt-to-llvm \
    --reconcile-unrealized-casts \
    -o "${MLIR_LLVM}"

echo "=== [3/4] Translating LLVM dialect MLIR -> LLVM IR (.ll) ==="
mlir-translate --mlir-to-llvmir "${MLIR_LLVM}" -o "${LLVM_IR}"

echo "=== [4/4] Compiling LLVM IR -> shared library (.so) ==="
clang -shared -fPIC \
    -O2 \
    "${LLVM_IR}" \
    -o "${OUTPUT_SO}" \
    -lm

echo "=== Build complete: ${OUTPUT_SO} ==="