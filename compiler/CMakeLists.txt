cmake_minimum_required(VERSION 3.15.0)
project(custom-opt)

# 1. Find the MLIR package
find_package(MLIR REQUIRED CONFIG)

message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)
set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

# 2. Define the executable and its source files
add_llvm_executable(custom-opt
  custom-opt.cpp
  ConvertMatmulToBlas.cpp
  AddOnePattern.cpp
  PARTIAL_SOURCES_INTENDED
)

# 3. Link all required MLIR Dialects and Conversion libraries
target_link_libraries(custom-opt PRIVATE
  MLIRRegisterAllDialects
  MLIRRegisterAllPasses
  MLIRIR
  MLIROptLib
  MLIRPass
  MLIRTransforms
  MLIRLinalgDialect
  MLIRLLVMDialect
  MLIRMemRefDialect
  MLIRArithDialect
  MLIRFuncDialect
  MLIRBufferizationDialect
  MLIRBufferizationPipelines
  MLIRBufferizationTransforms
  MLIRTensorToLinalg
  MLIRMemRefToLLVM
  MLIRAffineToStandard
  MLIRSCFToControlFlow
  MLIRArithToLLVM
  MLIRMathToLLVM
  MLIRMathToLibm
  MLIRControlFlowToLLVM
  MLIRFuncToLLVM
  MLIRReconcileUnrealizedCasts
)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})