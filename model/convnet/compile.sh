#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
source "$SCRIPT_DIR/../../config.sh"

LIB_PATH="$TORCH_MLIR_BUILD/lib"

sed -i 's/func.func @main\(.*\) {/func.func @forward\1 attributes {llvm.emit_c_interface} {/g' $BUILD/convnet.mlir

echo "1. Lowering MLIR to LLVM Dialect..."
$MLIR_OPT build/convnet.mlir \
  -empty-tensor-to-alloc-tensor \
  -one-shot-bufferize="bufferize-function-boundaries=1" \
  -convert-linalg-to-loops \
  -lower-affine \
  -convert-scf-to-cf \
  -convert-cf-to-llvm \
  -convert-math-to-llvm \
  -convert-math-to-libm \
  -convert-arith-to-llvm \
  -convert-vector-to-llvm \
  -expand-strided-metadata \
  -finalize-memref-to-llvm \
  -convert-func-to-llvm="use-bare-ptr-memref-call-conv=0" \
  -reconcile-unrealized-casts \
  -o build/convnet_llvm.mlir

echo "2. Translating to LLVM IR..."
$TRANSLATE -mlir-to-llvmir  build/convnet_llvm.mlir -o  build/convnet.ll

echo "3. Compile LLVM IR to .o"
$LLC -O3 -filetype=obj -relocation-model=pic build/convnet.ll -o  build/convnet.o

echo "4. Compiling run.cpp..."
$CLANG -O3 run.cpp  build/convnet.o -o  build/run \
  -L/home/anhtu/torch-mlir/build/lib \
  -lmlir_c_runner_utils \
  -Wl,-rpath,/home/anhtu/torch-mlir/build/lib

echo "Done!"