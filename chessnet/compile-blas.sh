#!/bin/bash
set -e

CUSTOM_OPT="../custom-opt/build/bin/custom-opt"
TORCH_MLIR_OPT="/home/anhtu/torch-mlir/build/bin/torch-mlir-opt"
MLIR_OPT="/home/anhtu/torch-mlir/build/bin/mlir-opt"
TRANSLATE="/home/anhtu/torch-mlir/build/bin/mlir-translate"
LLC="/home/anhtu/torch-mlir/build/bin/llc"
CLANG="/usr/bin/clang++"
LIB_PATH="/home/anhtu/torch-mlir/build/lib"
BUILD="./build"

$CUSTOM_OPT $BUILD/chessnet.mlir \
  -pass-pipeline="builtin.module(linalg-to-bufferization,bufferization-to-llvm)" \
  -o $BUILD/chessnet_llvm.mlir

$TRANSLATE -mlir-to-llvmir $BUILD/chessnet_llvm.mlir -o $BUILD/chessnet.ll

$LLC -O3 -filetype=obj -relocation-model=pic $BUILD/chessnet.ll -o $BUILD/chessnet.o

$CLANG -O3 main.cpp $BUILD/chessnet.o -o $BUILD/run_blas \
  -L$LIB_PATH \
  -lmlir_c_runner_utils \
  -lopenblas \
  -Wl,-rpath,$LIB_PATH

echo "Compilation successful."