#!/bin/bash
set -e

TORCH_MLIR_BUILD="/home/anhtu/torch-mlir/build"
TRANSLATE="$TORCH_MLIR_BUILD/bin/mlir-translate"
LLC="$TORCH_MLIR_BUILD/bin/llc"
CLANG="/usr/bin/clang++"
LIB_PATH="$TORCH_MLIR_BUILD/lib"
CUSTOM_OPT="../../custom-opt/build/bin/custom-opt"

MODEL_NAME="mobilenet"
BUILD_DIR="build"

echo "Compiling ${MODEL_NAME}..."

echo "- Translate to LLVM MLIR"
$CUSTOM_OPT ${BUILD_DIR}/${MODEL_NAME}.mlir \
  -pass-pipeline="builtin.module(linalg-to-bufferization,bufferization-to-llvm)" \
  -o ${BUILD_DIR}/${MODEL_NAME}_llvm.mlir

echo "- Translate to LLVM IR"
$TRANSLATE -mlir-to-llvmir ${BUILD_DIR}/${MODEL_NAME}_llvm.mlir -o ${BUILD_DIR}/${MODEL_NAME}.ll

echo "- Compile LLVM IR to .o"
$LLC -O3 -filetype=obj -relocation-model=pic ${BUILD_DIR}/${MODEL_NAME}.ll -o ${BUILD_DIR}/${MODEL_NAME}.o

echo "- Compile run.cpp"
$CLANG -O3 run.cpp ${BUILD_DIR}/${MODEL_NAME}.o -o ${BUILD_DIR}/run \
  -L$LIB_PATH \
  -L/usr/lib/x86_64-linux-gnu \
  -lmlir_c_runner_utils \
  -lopenblas \
  -Wl,-rpath,$LIB_PATH

echo "Done!"