#!/bin/bash
set -e

TORCH_MLIR_BUILD="/home/anhtu/torch-mlir/build"
MLIR_OPT="$TORCH_MLIR_BUILD/bin/mlir-opt"
TRANSLATE="$TORCH_MLIR_BUILD/bin/mlir-translate"
LLC="$TORCH_MLIR_BUILD/bin/llc"
CLANG="/usr/bin/clang++"
LIB_PATH="$TORCH_MLIR_BUILD/lib"
BUILD="./build"

sed -i 's/func.func @main\(.*\) {/func.func @forward\1 attributes {llvm.emit_c_interface} {/g' $BUILD/convnet.mlir

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

$TRANSLATE -mlir-to-llvmir  $BUILD/convnet_llvm.mlir -o  $BUILD/convnet.ll
$LLC -O3 -filetype=obj -relocation-model=pic  $BUILD/convnet.ll -o  $BUILD/convnet.o

$CLANG -O3 main.cpp  $BUILD/convnet.o -o  $BUILD/run \
  -L/home/anhtu/torch-mlir/build/lib \
  -lmlir_c_runner_utils \
  -Wl,-rpath,/home/anhtu/torch-mlir/build/lib

echo "Compilation successful."