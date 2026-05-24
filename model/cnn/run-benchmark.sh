#!/bin/bash

export OPENBLAS_NUM_THREADS=1

echo "Running Naive MLIR..."
taskset -c 0 ./build/run

echo "Running OpenBLAS..."
taskset -c 0 ./build/run-blas

echo "Running OpenBLAS..."
taskset -c 0 ./build/run-blas-fuse