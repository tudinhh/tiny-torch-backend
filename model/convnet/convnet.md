## About the Model
This targets a standard Convolutional Neural Network.

## Custom pass: OpenBLAS
The compilation process leverages a custom MLIR optimization pass to replace standard matrix computations with high-performance OpenBLAS calls.

### Tool Entry Point (custom-opt.cpp)
This file defines and registers the custom MLIR optimizer, bundling transformations into automated pipelines.

**Pipeline Definition**: Registers two primary pipelines (`linalg-to-bufferization` and `bufferization-to-llvm`).

**Sequential Execution**: Passes are appended using `manager.addPass()` and execute in strict top-down order.

**Single Responsibility**: Each pass is narrowly scoped to a single transformation. Complex behavior is achieved by chaining multiple small passes.

**Interleaved Cleanup**: Simplification passes, such as CanonicalizerPass and CSEPass, run between major transformations to remove dead code and optimize the IR.

**Pipeline Advantage**: Defining passes in C++ ensures execution order and is heavily preferred over long sequences of manual CLI flags.

### Custom Pass (ConvertMatmulToBlas.cpp)
This file defines the conversion logic to replace `linalg.matmul` operations with OpenBLAS calls, utilizing the standard "Match and Rewrite" paradigm:

**Match**: Intercepts the target operation and verifies it meets the necessary constraints (e.g., correct tensor shapes and data types).

**Rewrite (Extract & Prepare)**: Extracts data pointers, generates necessary arguments, and creates lower-level operations (like an llvm.call) using the `Rewriter`.

**Replace/Erase**: Deletes the original operation from the compiler graph.

## Runtime
This time, a profiling code is added into `run.cpp` to compare the run time between the code with and without OpenBLAS:
- w/ OpenBLAS: 124 ms
- w/o OpenBLAS: 144 ms

## Pop-up Questions
No question.