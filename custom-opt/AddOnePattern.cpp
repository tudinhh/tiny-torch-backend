#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include <memory>

namespace mlir {
namespace tutorial {

struct ExperimentAddOnePattern : public OpRewritePattern<arith::AddIOp> {
  using OpRewritePattern<arith::AddIOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::AddIOp addOp, PatternRewriter &rewriter) const override {
    // 1. Prevent infinite recursion. If we already processed this, skip it.
    if (addOp->hasAttr("add_one_done")) return failure();

    Location loc = addOp.getLoc();
    Value lhs = addOp.getLhs();
    Value rhs = addOp.getRhs();
    Type type = addOp.getType();

    if (!isa<IntegerType>(type)) return failure();

    // 2. Prepare the "+ 1" constant
    IntegerAttr oneAttr = rewriter.getIntegerAttr(type, 1);
    Value constantOne = rewriter.create<arith::ConstantOp>(loc, type, cast<TypedAttr>(oneAttr));

    // 3. Recreate original A + B and mark it as done
    Value newAdd = rewriter.create<arith::AddIOp>(loc, lhs, rhs);
    newAdd.getDefiningOp()->setAttr("add_one_done", rewriter.getUnitAttr());

    // 4. Create (A + B) + 1 and mark it as done
    Value addOne = rewriter.create<arith::AddIOp>(loc, newAdd, constantOne);
    addOne.getDefiningOp()->setAttr("add_one_done", rewriter.getUnitAttr());

    // 5. Replace and erase the original op
    rewriter.replaceOp(addOp, addOne);
    
    return success();
  }
};

struct ExperimentAddOnePass : public PassWrapper<ExperimentAddOnePass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ExperimentAddOnePass)
  StringRef getArgument() const final { return "experiment-add-one"; }
  StringRef getDescription() const final { return "Replaces A+B with A+B+1 for testing."; }

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<ExperimentAddOnePattern>(&getContext());

    // Execute standard greedy rewrite driver on the current function
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace tutorial
} // namespace mlir

std::unique_ptr<mlir::Pass> createExperimentAddOnePass() {
  return std::make_unique<mlir::tutorial::ExperimentAddOnePass>();
}