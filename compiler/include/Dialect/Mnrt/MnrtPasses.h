#pragma once
#include "mlir/Pass/Pass.h"
#include <memory>

namespace mnrt {
    std::unique_ptr<mlir::Pass> createConvertMemRefToMnrtPass();

    #define GEN_PASS_DECL
    #define GEN_PASS_REGISTRATION // Add this line
    #include "MnrtPasses.h.inc"
}