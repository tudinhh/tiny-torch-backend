include "mlir/Pass/PassBase.td"

def ConvertMemRefToMnrt : Pass<"convert-memref-to-mnrt"> {
  let summary = "Lower MemRef calls to Mnrt dispatch operations.";
  let constructor = "mnrt::createConvertMemRefToMnrtPass()";
  let dependentDialects = ["mnrt::MnrtDialect"];
}