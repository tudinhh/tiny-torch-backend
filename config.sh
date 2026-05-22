#!/bin/bash



# export LLVM_BUILD_DIR="/path/to/llvm-project/build"
export TORCH_MLIR_DIR="/home/anhtu/torch-mlir"

export PYTHONPATH=/home/anhtu/torch-mlir/build/tools/torch-mlir/python_packages/torch_mlir:$PYTHONPATH

export MLIR_OPT="$TORCH_MLIR_DIR/build/bin/mlir-opt"
export TRANSLATE="$TORCH_MLIR_DIR/build/bin/mlir-translate"
export LLC="$TORCH_MLIR_DIR/build/bin/llc"
export CLANG="/usr/bin/clang++"