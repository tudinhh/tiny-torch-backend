#!/bin/bash
set -e

mkdir -p build
cd build

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
source "$SCRIPT_DIR/../../config.sh"

cmake -G Ninja \
  -DMLIR_DIR=$TORCH_MLIR_DIR/build/lib/cmake/mlir \
  -DLLVM_DIR=$TORCH_MLIR_DIR/build/lib/cmake/llvm \
  ..

ninja