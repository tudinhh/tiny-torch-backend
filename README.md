# tiny-torch-backend

A minimal, custom MLIR backend pipeline that lowers PyTorch models directly to native OpenBLAS execution. 

This project demonstrates how to write a custom MLIR pass to intercept `linalg.matmul` operations and swap them with optimized C-calls to OpenBLAS, completely bypassing LLVM's default nested loops.

## Prerequisites
* LLVM & MLIR
* Torch-MLIR
* OpenBLAS (`sudo apt-get install libopenblas-dev`)

## Project Structure
* `custom-opt/`: Contains the C++ source for the custom MLIR pass (`ConvertMatmulToBlas.cpp`).
* `compile.sh`: The progressive lowering pipeline that translates `.mlir` to LLVM IR, compiles to an object file, and links the native libraries.
* `main.cpp`: The C++ runner that initializes the `MemRefDescriptor`, calls the compiled MLIR function, and handles I/O.
