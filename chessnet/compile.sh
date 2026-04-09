#!/bin/bash
set -e

# Paths
TORCH_MLIR_OPT="/home/anhtu/torch-mlir/build/bin/torch-mlir-opt"
MLIR_OPT="/home/anhtu/torch-mlir/build/bin/mlir-opt"
TRANSLATE="/home/anhtu/torch-mlir/build/bin/mlir-translate"
LLC="/home/anhtu/torch-mlir/build/bin/llc"
CLANG="/usr/bin/clang++"
LIB_PATH="/home/anhtu/torch-mlir/build/lib"
BUILD="./build"
#!/bin/bash
set -e

# 1. Inject the emit_c_interface attribute into the forward function
# This ensures the _mlir_ciface_ wrapper is generated.
sed -i 's/func.func @main\(.*\) {/func.func @forward\1 attributes {llvm.emit_c_interface} {/g' $BUILD/chessnet.mlir

# 2. Lower Linalg-on-Tensors to LLVM Dialect
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
  -o  build/chessnet_llvm.mlir

$TRANSLATE -mlir-to-llvmir  $BUILD/chessnet_llvm.mlir -o  $BUILD/chessnet.ll
$LLC -filetype=obj -relocation-model=pic  $BUILD/chessnet.ll -o  $BUILD/chessnet.o

clang++ main.cpp  $BUILD/chessnet.o -o  $BUILD/chessnet_runner \
  -L/home/anhtu/torch-mlir/build/lib \
  -lmlir_c_runner_utils \
  -Wl,-rpath,/home/anhtu/torch-mlir/build/lib

echo "Compilation successful. Run with ./chessnet_runner"