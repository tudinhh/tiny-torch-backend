#!/bin/bash
set -e

mkdir -p build
cd build

TORCH_MLIR="/home/anhtu/torch-mlir"

cmake -G Ninja \
  -DMLIR_DIR=$TORCH_MLIR/build/lib/cmake/mlir \
  -DLLVM_DIR=$TORCH_MLIR/build/lib/cmake/llvm \
  ..

ninja