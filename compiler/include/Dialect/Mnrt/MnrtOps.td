// include standard MLIR traits and interfaces
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

// ===----------------------------------------------------------------------===//
// Mnrt Dialect Definition
// ===----------------------------------------------------------------------===//

def Mnrt_Dialect : Dialect {
  let name = "mnrt";
  let summary = "Mini-Runtime hardware abstraction and dispatch dialect.";
  let description = [{
    The `mnrt` dialect models explicit hardware command buffers, memory hierarchies,
    and kernel dispatch.
  }];
  let cppNamespace = "::mnrt";
}

// Base class for all mnrt operations
class Mnrt_Op<string mnemonic, list<Trait> traits = []> :
    Op<Mnrt_Dialect, mnemonic, traits>;

// ===----------------------------------------------------------------------===//
// Mnrt Operations
// ===----------------------------------------------------------------------===//

def Mnrt_AllocOp : Mnrt_Op<"alloc"> {
  let summary = "Allocate memory on a specific hardware tier.";
  let description = [{
    Allocates a buffer. The `memory_space` attribute determines the hardware
    tier (e.g., 0 for Host DRAM, 1 for Device SRAM).
  }];
  
  let arguments = (ins OptionalAttr<I32Attr>:$memory_space);
  let results = (outs AnyMemRef:$result);

  let assemblyFormat = "attr-dict `:` type($result)";
}

def Mnrt_DmaCopyOp : Mnrt_Op<"dma_copy"> {
  let summary = "Asynchronous DMA copy between memory spaces.";
  let description = [{
    Moves data between host and device, or between device tiers.
  }];

  let arguments = (ins AnyMemRef:$src, AnyMemRef:$dst);
  
  let assemblyFormat = "$src `,` $dst attr-dict `:` type($src) `to` type($dst)";
}

def Mnrt_DispatchOp : Mnrt_Op<"dispatch", [AttrSizedOperandSegments]> {
  let summary = "Queue a compute kernel for execution.";
  let description = [{
    Dispatches a compiled kernel to the device command queue. It explicitly 
    separates input and output buffers to match the runtime ABI.
  }];

  let arguments = (ins FlatSymbolRefAttr:$callee,
                       Variadic<AnyMemRef>:$inputs,
                       Variadic<AnyMemRef>:$outputs);

  let assemblyFormat = [{
    $callee `(` $inputs `)` `outs` `(` $outputs `)` attr-dict 
    `:` `(` type($inputs) `)` `->` type($outputs)
  }];
}