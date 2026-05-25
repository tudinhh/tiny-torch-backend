#!/bin/bash
set -e 

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
source "$SCRIPT_DIR/../../config.sh"

CUSTOM_OPT="../../custom-opt/build/bin/custom-opt"

echo "- Apply MLIR transformations"
$CUSTOM_OPT build/add_linalg.mlir \
  -empty-tensor-to-alloc-tensor \
  -one-shot-bufferize="bufferize-function-boundaries=1" \
  -convert-linalg-to-loops \
  -experiment-add-one \
  -lower-affine \
  -convert-scf-to-cf \
  -expand-strided-metadata \
  -llvm-request-c-wrappers \
  -convert-math-to-llvm \
  -convert-arith-to-llvm \
  -finalize-memref-to-llvm \
  -convert-func-to-llvm \
  -convert-cf-to-llvm \
  -reconcile-unrealized-casts \
  -o build/add_llvm.mlir

echo "- Translate MLIR to LLVM IR"
$TRANSLATE -mlir-to-llvmir build/add_llvm.mlir > build/add.ll

echo "- Compile LLVM IR to .o"
$LLC -filetype=obj build/add.ll -o build/add.o

echo "- Compile run.cpp"
$CLANG -O3 run.cpp build/add.o -o build/run \
  -L$TORCH_MLIR_DIR/build/lib \
  -lmlir_c_runner_utils \
  -Wl,-rpath,$TORCH_MLIR_DIR/build/lib

echo "Done!"