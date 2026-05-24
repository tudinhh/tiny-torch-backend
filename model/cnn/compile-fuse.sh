#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
source "$SCRIPT_DIR/../../config.sh"

CUSTOM_OPT="../../custom-opt/build/bin/custom-opt"
LIB_PATH="$TORCH_MLIR_DIR/build/lib"
BUILD="./build"

echo "1. Lowering MLIR to LLVM Dialect..."
$CUSTOM_OPT build/cnn.mlir \
  -pass-pipeline="builtin.module(linalg-to-bufferization,bufferization-to-llvm)" \
  -o build/cnn_llvm.mlir

echo "2. Translating to LLVM IR..."
$TRANSLATE -mlir-to-llvmir build/cnn_llvm.mlir -o build/cnn.ll

echo "3. Compile LLVM IR to .o"
$LLC -O3 -filetype=obj -relocation-model=pic build/cnn.ll -o build/cnn.o

echo "4. Compiling run.cpp..."
$CLANG -O3 run.cpp build/cnn.o -o build/run-blas-fuse \
  -L$LIB_PATH \
  -lmlir_c_runner_utils \
  -lopenblas \
  -Wl,-rpath,$LIB_PATH

echo "Done!"