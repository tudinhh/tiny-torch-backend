#!/bin/bash
set -e

TORCH_MLIR_OPT="/home/anhtu/torch-mlir/build/bin/torch-mlir-opt"
MLIR_OPT="/home/anhtu/torch-mlir/build/bin/mlir-opt"
TRANSLATE="/home/anhtu/torch-mlir/build/bin/mlir-translate"
LLC="/home/anhtu/torch-mlir/build/bin/llc"
CLANG="/usr/bin/clang++"
LIB_PATH="/home/anhtu/torch-mlir/build/lib"
BUILD="./build"

set -e

sed -i 's/func.func @main\(.*\) {/func.func @forward\1 attributes {llvm.emit_c_interface} {/g' $BUILD/chessnet.mlir

$MLIR_OPT build/chessnet.mlir \
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
  -o build/chessnet_llvm.mlir

$TRANSLATE -mlir-to-llvmir  $BUILD/chessnet_llvm.mlir -o  $BUILD/chessnet.ll
$LLC -O3 -filetype=obj -relocation-model=pic  $BUILD/chessnet.ll -o  $BUILD/chessnet.o

clang++ -O3 main.cpp  $BUILD/chessnet.o -o  $BUILD/run \
  -L/home/anhtu/torch-mlir/build/lib \
  -lmlir_c_runner_utils \
  -Wl,-rpath,/home/anhtu/torch-mlir/build/lib

echo "Compilation successful."