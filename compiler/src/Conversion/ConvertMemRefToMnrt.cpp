#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "Dialect/Mnrt/MnrtDialect.h"
#include "Dialect/Mnrt/MnrtOps.h"
#include "Dialect/Mnrt/MnrtPasses.h"

namespace mnrt {
#define GEN_PASS_DEF_CONVERTMEMREFTOMNRT
#include "MnrtPasses.h.inc"
}

using namespace mlir;
using namespace mnrt;

namespace {

struct DispatchConversion : public OpConversionPattern<linalg::GenericOp> {
  using OpConversionPattern<linalg::GenericOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(linalg::GenericOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    StringAttr targetAttr = rewriter.getStringAttr("cpu");

    rewriter.replaceOpWithNewOp<DispatchOp>(
        op,
        TypeRange{},
        rewriter.getStringAttr("generic_kernel"),
        adaptor.getInputs(),
        adaptor.getOutputs()
    );

    return success();
  }
};

struct ConvertMemRefToMnrtPass : public mnrt::impl::ConvertMemRefToMnrtBase<ConvertMemRefToMnrtPass> {
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ConversionTarget target(*context);

    target.addLegalDialect<MnrtDialect>();
    target.addIllegalOp<linalg::GenericOp>();

    RewritePatternSet patterns(context);
    patterns.add<DispatchConversion>(context);

    if (failed(applyPartialConversion(getOperation(), target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

}

namespace mnrt {
std::unique_ptr<mlir::Pass> createConvertMemRefToMnrtPass() {
  return std::make_unique<ConvertMemRefToMnrtPass>();
}
}