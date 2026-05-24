#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
source "$SCRIPT_DIR/../../config.sh"

CUSTOM_OPT="../../custom-opt/build/bin/custom-opt"
LIB_PATH="$TORCH_MLIR_DIR/build/lib"
BUILD="./build"

echo "1. Lowering MLIR to LLVM Dialect..."
$CUSTOM_OPT build/convnet.mlir \
  -pass-pipeline="builtin.module(linalg-to-bufferization,bufferization-to-llvm)" \
  -o build/convnet_llvm.mlir

echo "2. Translating to LLVM IR..."
$TRANSLATE -mlir-to-llvmir build/convnet_llvm.mlir -o build/convnet.ll

echo "3. Compile LLVM IR to .o"
$LLC -O3 -filetype=obj -relocation-model=pic build/convnet.ll -o build/convnet.o

echo "4. Compiling run.cpp..."
$CLANG -O3 run.cpp build/convnet.o -o build/run-blas \
  -L$LIB_PATH \
  -lmlir_c_runner_utils \
  -lopenblas \
  -Wl,-rpath,$LIB_PATH

echo "Done!"