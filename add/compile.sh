#!/bin/bash
set -e 

MLIR_OPT="/home/anhtu/torch-mlir/build/bin/mlir-opt"
TRANSLATE="/home/anhtu/torch-mlir/build/bin/mlir-translate"
LLC="/home/anhtu/torch-mlir/build/bin/llc"

echo "1. Lowering MLIR to LLVM Dialect..."
$MLIR_OPT add_linalg.mlir \
  -empty-tensor-to-alloc-tensor \
  -one-shot-bufferize="bufferize-function-boundaries=1" \
  -convert-linalg-to-loops \
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
  -o add_llvm.mlir

echo "2. Translating to LLVM IR..."
$TRANSLATE -mlir-to-llvmir add_llvm.mlir > add.ll

echo "3. Compiling Object File..."
$LLC -filetype=obj add.ll -o add.o

echo "Success: Created add.o"