#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"

namespace mlir {
namespace tutorial {

struct MatmulOpToBlasLibraryCall : public ConversionPattern {
  explicit MatmulOpToBlasLibraryCall(MLIRContext *context, TypeConverter &converter)
      : ConversionPattern(converter, linalg::MatmulOp::getOperationName(), 1, context) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                                ConversionPatternRewriter &rewriter) const override {
    auto matmulOp = cast<linalg::MatmulOp>(op);
    Location loc = matmulOp.getLoc();
    
    // Validation and type checking
    auto inputs = matmulOp.getDpsInputs();
    auto outputs = matmulOp.getDpsInits();
    if (inputs.size() != 2 || outputs.size() != 1) return failure();

    Value originalLhs = inputs[0];
    Value originalRhs = inputs[1];
    Value originalOutput = outputs[0];

    auto lhsType = dyn_cast<MemRefType>(originalLhs.getType());
    auto rhsType = dyn_cast<MemRefType>(originalRhs.getType());
    auto outputType = dyn_cast<MemRefType>(originalOutput.getType());

    if (!lhsType || !rhsType || !outputType || lhsType.getRank() != 2 ||
        rhsType.getRank() != 2 || outputType.getRank() != 2 ||
        !lhsType.getElementType().isF32()) {
      return failure();
    }

    Value lhs = operands[0];
    Value rhs = operands[1];
    Value output = operands[2];

    auto lhsShape = lhsType.getShape();
    auto rhsShape = rhsType.getShape();
    auto outputShape = outputType.getShape();

    auto i32Type = IntegerType::get(rewriter.getContext(), 32);
    auto f32Type = Float32Type::get(rewriter.getContext());
    auto ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());

    ModuleOp module = matmulOp->getParentOfType<ModuleOp>();
    LLVM::LLVMFuncOp sgemmFunc = getOrCreateSgemmFunc(module, rewriter);

    Value order = rewriter.create<LLVM::ConstantOp>(loc, i32Type, rewriter.getI32IntegerAttr(101));
    Value transA = rewriter.create<LLVM::ConstantOp>(loc, i32Type, rewriter.getI32IntegerAttr(111));
    Value transB = rewriter.create<LLVM::ConstantOp>(loc, i32Type, rewriter.getI32IntegerAttr(111));
    
    Value M, N, K, ldA, ldB, ldC;

    if (lhsShape[0] != ShapedType::kDynamic) {
      M = rewriter.create<LLVM::ConstantOp>(loc, i32Type, rewriter.getI32IntegerAttr(lhsShape[0]));
    } else {
      Value dimM = rewriter.create<memref::DimOp>(loc, originalLhs, 0);
      M = rewriter.create<arith::IndexCastOp>(loc, i32Type, dimM);
    }

    if (rhsShape[1] != ShapedType::kDynamic) {
      N = rewriter.create<LLVM::ConstantOp>(loc, i32Type, rewriter.getI32IntegerAttr(rhsShape[1]));
    } else {
      Value dimN = rewriter.create<memref::DimOp>(loc, originalRhs, 1);
      N = rewriter.create<arith::IndexCastOp>(loc, i32Type, dimN);
    }

    if (lhsShape[1] != ShapedType::kDynamic) {
      K = rewriter.create<LLVM::ConstantOp>(loc, i32Type, rewriter.getI32IntegerAttr(lhsShape[1]));
      ldA = rewriter.create<LLVM::ConstantOp>(loc, i32Type, rewriter.getI32IntegerAttr(lhsShape[1]));
    } else {
      Value dimK = rewriter.create<memref::DimOp>(loc, originalLhs, 1);
      K = rewriter.create<arith::IndexCastOp>(loc, i32Type, dimK);
      ldA = K;
    }

    if (rhsShape[1] != ShapedType::kDynamic) {
      ldB = rewriter.create<LLVM::ConstantOp>(loc, i32Type, rewriter.getI32IntegerAttr(rhsShape[1]));
    } else {
      Value dimLdB = rewriter.create<memref::DimOp>(loc, originalRhs, 1);
      ldB = rewriter.create<arith::IndexCastOp>(loc, i32Type, dimLdB);
    }

    if (outputShape[1] != ShapedType::kDynamic) {
      ldC = rewriter.create<LLVM::ConstantOp>(loc, i32Type, rewriter.getI32IntegerAttr(outputShape[1]));
    } else {
      Value dimLdC = rewriter.create<memref::DimOp>(loc, originalOutput, 1);
      ldC = rewriter.create<arith::IndexCastOp>(loc, i32Type, dimLdC);
    }

    // Create alpha and beta for the BLAS equation
    Value alpha = rewriter.create<LLVM::ConstantOp>(loc, f32Type, rewriter.getF32FloatAttr(1.0));
    Value beta = rewriter.create<LLVM::ConstantOp>(loc, f32Type, rewriter.getF32FloatAttr(0.0));

    // Extract raw pointers from MemRefs
    Value lhsPtr = rewriter.create<LLVM::ExtractValueOp>(loc, lhs, ArrayRef<int64_t>{1});
    Value rhsPtr = rewriter.create<LLVM::ExtractValueOp>(loc, rhs, ArrayRef<int64_t>{1});
    Value outputPtr = rewriter.create<LLVM::ExtractValueOp>(loc, output, ArrayRef<int64_t>{1});

    // Group 14 arguments in an exact order required by cblas_gemm
    SmallVector<Value> args = {order, transA, transB, M, N, K, alpha, lhsPtr, ldA, rhsPtr, ldB, beta, outputPtr, ldC};
    
    // Add instruction to call cblas_gemm
    rewriter.create<LLVM::CallOp>(loc, sgemmFunc, args);

    // Erase linalg.matmul from IR graph
    rewriter.eraseOp(matmulOp);
    return success();
  }

private:
  LLVM::LLVMFuncOp getOrCreateSgemmFunc(ModuleOp module, PatternRewriter &rewriter) const {
    const StringRef funcName = "cblas_sgemm";
    if (auto existingFunc = module.lookupSymbol<LLVM::LLVMFuncOp>(funcName)) return existingFunc;

    auto i32Type = IntegerType::get(rewriter.getContext(), 32);
    auto f32Type = Float32Type::get(rewriter.getContext());
    auto ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());

    SmallVector<Type> argTypes = {i32Type, i32Type, i32Type, i32Type, i32Type, i32Type, f32Type, ptrType, i32Type, ptrType, i32Type, f32Type, ptrType, i32Type};
    auto funcType = LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(rewriter.getContext()), argTypes);

    PatternRewriter::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(module.getBody());
    auto sgemmFunc = rewriter.create<LLVM::LLVMFuncOp>(rewriter.getUnknownLoc(), funcName, funcType);
    sgemmFunc.setPrivate();
    return sgemmFunc;
  }
};

struct ConvertMatmulToBlasLibraryCallPass : public PassWrapper<ConvertMatmulToBlasLibraryCallPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertMatmulToBlasLibraryCallPass)
  StringRef getArgument() const final { return "convert-matmul-to-blas"; }
  StringRef getDescription() const final { return "Convert linalg.matmul to OpenBLAS calls."; }

  void runOnOperation() override {
    ConversionTarget target(getContext());
    target.addLegalDialect<func::FuncDialect, LLVM::LLVMDialect, memref::MemRefDialect, arith::ArithDialect>();
    target.addIllegalOp<linalg::MatmulOp>();

    RewritePatternSet patterns(&getContext());
    LLVMTypeConverter typeConverter(&getContext());
    patterns.add<MatmulOpToBlasLibraryCall>(patterns.getContext(), typeConverter);

    if (failed(applyPartialConversion(getOperation(), target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace tutorial
} // namespace mlir

std::unique_ptr<mlir::Pass> createConvertMatmulToBlasLibraryCallPass() {
  return std::make_unique<mlir::tutorial::ConvertMatmulToBlasLibraryCallPass>();
}