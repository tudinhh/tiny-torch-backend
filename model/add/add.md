## About the model
A simple Pytorch model that adds two tensors.

## Compilation
### Capture the model and translate to MLIR
The model is captured by Pytorch. Then, it is exported into `torch` dialect.
```Python
module_torch = torchscript.compile(
    model, 
    (a, b), 
    output_type="torch"
)
```
At this point, the output `add_torch.mlir` file still shows the high-level semantics of the model with ATen operations.

The second option is translated into `linalg` dialect but on the tensor type. After this, the IR in `add_linalg.mlir` starts loosing the high-level architecture of the model.

### Further lowering
Once the IR reach `linalg` dialect, the compiler performs a series of transformations. Eventually, the MLIR is converted into LLVM IR in `add.ll`.

Then, the file is compiled into an object file, which then is used to build an executable.

To summarize, the model undergoes a series of transformations: PyTorch model → torch dialect → linalg dialect  → LLVM IR → executable. By the time it reaches the LLVM IR stage, it has lost all high-level architectural semantics, leaving only low-level instructions such as load, store, and add.

### Runtime
The run file does:
- Allocate the memory for input and output tensors
- Wrap the arrays into `MemRefDescriptor` to pass into MLIR code
- Call `_mlir_ciface_forward` to run the model.
So, basically, this run time allocates the memory, packs them it an appropriate way, then calls the model.
## Pop-up questions
>This experiment uses `torch-mlir` to compile a model. What is the default compilation of Pytorch?

Pytorch uses TorchInductor as the primary compiler backend. TorchInductor lowers the model graph then generates Triton kernels. Finally, it emits machine code.

>A simple runtime does: allocate -> organize -> execute. How about the production-grade runtimes like IREE and ONNX Runtime?



