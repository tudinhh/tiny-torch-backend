#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Conversion/TensorToLinalg/TensorToLinalgPass.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Bufferization/Pipelines/Passes.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/MathToLibm/MathToLibm.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"

#include "mlir/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Conversion/TensorToLinalg/TensorToLinalgPass.h"
#include "mlir/Conversion/LinalgToStandard/LinalgToStandard.h"

#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"

std::unique_ptr<mlir::Pass> createConvertMatmulToBlasLibraryCallPass();
std::unique_ptr<mlir::Pass> createExperimentAddOnePass();

// Define a pipeline with many passes: linalgToBufferizationPipeline
void linalgToBufferizationPipelineBuilder(mlir::OpPassManager &manager) {
  manager.addPass(mlir::createCanonicalizerPass());
  manager.addPass(mlir::createConvertTensorToLinalgPass());

  mlir::bufferization::OneShotBufferizePassOptions bufferizationOptions;
  bufferizationOptions.bufferizeFunctionBoundaries = true;
  manager.addPass(mlir::bufferization::createOneShotBufferizePass(bufferizationOptions));
  
  mlir::bufferization::BufferDeallocationPipelineOptions deallocationOptions;
  mlir::bufferization::buildBufferDeallocationPipeline(manager, deallocationOptions);
}

// Define a pipeline with many passes: BufferizationToLLVMPipeline
void BufferizationToLLVMPipelineBuilder(mlir::OpPassManager &manager) {
  // manager.addPass(createConvertMatmulToBlasLibraryCallPass());
  
  manager.addPass(mlir::createConvertLinalgToLoopsPass());
  manager.addPass(mlir::memref::createExpandStridedMetadataPass());
  manager.addPass(mlir::createLowerAffinePass());
  manager.addPass(mlir::createSCFToControlFlowPass());
  
  manager.addPass(mlir::createArithToLLVMConversionPass());
  manager.addPass(mlir::createConvertMathToLLVMPass());
  manager.addPass(mlir::createConvertMathToLibmPass());
  manager.addPass(mlir::createConvertControlFlowToLLVMPass());
  manager.addPass(mlir::createFinalizeMemRefToLLVMConversionPass());
  manager.addPass(mlir::createConvertFuncToLLVMPass());
  manager.addPass(mlir::createReconcileUnrealizedCastsPass());
  
  manager.addPass(mlir::createCanonicalizerPass());
  manager.addPass(mlir::createSCCPPass());
  manager.addPass(mlir::createCSEPass());
  manager.addPass(mlir::createSymbolDCEPass());
}

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  mlir::registerAllPasses();

  mlir::registerPass([]() {
        return createConvertMatmulToBlasLibraryCallPass();
  });

  mlir::registerPass([]() {
        return createExperimentAddOnePass();
  });

  mlir::PassPipelineRegistration<>(
      "linalg-to-bufferization", "Lower linalg to bufferization", linalgToBufferizationPipelineBuilder);

  mlir::PassPipelineRegistration<>(
      "bufferization-to-llvm", "Lower bufferized code to LLVM", BufferizationToLLVMPipelineBuilder);

  return mlir::asMainReturnCode(mlir::MlirOptMain(argc, argv, "Tutorial Pass Driver", registry));
}