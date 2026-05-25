#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
source "$SCRIPT_DIR/../../config.sh"

CUSTOM_OPT="../../custom-opt/build/bin/custom-opt"
LIB_PATH="$TORCH_MLIR_DIR/build/lib"

echo "- Apply MLIR transformations"
$CUSTOM_OPT build/cnn.mlir \
  -pass-pipeline="builtin.module(linalg-to-bufferization,convert-matmul-to-blas,bufferization-to-llvm)" \
  -o build/cnn_llvm.mlir

echo "- Translate MLIR to LLVM IR"
$TRANSLATE -mlir-to-llvmir build/cnn_llvm.mlir -o build/cnn.ll

echo "- Compile LLVM IR to .o"
$LLC -O3 -filetype=obj -relocation-model=pic build/cnn.ll -o build/cnn.o

echo "- Compile run.cpp"
$CLANG -O3 run.cpp build/cnn.o -o build/run-blas \
  -L$LIB_PATH \
  -lmlir_c_runner_utils \
  -lopenblas \
  -Wl,-rpath,$LIB_PATH

echo "Done!"