#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "Dialect/Mnrt/MnrtPasses.h"
#include "Dialect/Mnrt/MnrtDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"

namespace mnrt {
  void registerMnrtPasses();
}

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  
  registry.insert<
      mlir::arith::ArithDialect,
      mlir::func::FuncDialect,
      mlir::memref::MemRefDialect,
      mlir::scf::SCFDialect,
      mlir::LLVM::LLVMDialect,
      mlir::linalg::LinalgDialect,
      mnrt::MnrtDialect
  >();
  mnrt::registerMnrtPasses();
  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "Mnrt Optimizer Driver\n", registry));
}