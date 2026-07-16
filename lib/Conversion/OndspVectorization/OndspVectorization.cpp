#include "ondrix/Conversion/OndspVectorization/OndspVectorization.h"
#include "ondrix/Conversion/Utils/ReductionUtils.h"

#include "ondrix/Dialect/ondsp/IR/OndspDialect.h"
#include "ondrix/Dialect/ondsp/IR/OndspOps.h"
#include "ondrix/Dialect/ondsp/IR/OndspSemantics.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace ondrix {
#define GEN_PASS_DEF_VECTORIZEONDSPQ15MEMREFREDUCE
#include "ondrix/Conversion/Passes.h.inc"
} // namespace ondrix

using namespace mlir;

namespace {

bool isSupportedMemRefReduction(ondrix::ondsp::ReduceMacOp op) {
  auto accumulator = dyn_cast<ondrix::ondsp::AccType>(op.getInitial().getType());
  auto numeric = dyn_cast<ondrix::ondsp::FixedAttr>(op.getNumeric());
  auto lhsType = dyn_cast<MemRefType>(op.getLhs().getType());
  auto rhsType = dyn_cast<MemRefType>(op.getRhs().getType());
  return accumulator && numeric && op.getProduct() && lhsType && rhsType &&
         lhsType.getRank() == 1 && rhsType.getRank() == 1 &&
         lhsType.getElementType().isSignlessInteger(16) &&
         rhsType.getElementType().isSignlessInteger(16) && isLastMemrefDimUnitStride(lhsType) &&
         isLastMemrefDimUnitStride(rhsType) &&
         ondrix::ondsp::isSignedQ15I40Accumulator(accumulator) &&
         ondrix::ondsp::isSignedQ15(numeric) && ondrix::ondsp::isFullProduct(*op.getProduct());
}

class ReduceMacOpVectorization final : public OpConversionPattern<ondrix::ondsp::ReduceMacOp> {
public:
  ReduceMacOpVectorization(MLIRContext *context, int64_t vectorWidth)
      : OpConversionPattern(context), vectorWidth(vectorWidth) {}

  LogicalResult matchAndRewrite(ondrix::ondsp::ReduceMacOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    if (!isSupportedMemRefReduction(op))
      return failure();
    auto numeric = cast<ondrix::ondsp::FixedAttr>(op.getNumeric());

    FailureOr<ondrix::conversion::RankOneReductionBounds> bounds =
        ondrix::conversion::createRankOneMemRefReductionBounds(
            op, adaptor.getLhs(), adaptor.getRhs(), rewriter.getI16Type(),
            "Q15 memref vectorization", rewriter);
    if (failed(bounds))
      return failure();

    Location loc = op.getLoc();
    Value vectorStep = rewriter.create<arith::ConstantIndexOp>(loc, vectorWidth);
    Value remainder = rewriter.create<arith::RemUIOp>(loc, bounds->upperBound, vectorStep);
    Value vectorEnd = rewriter.create<arith::SubIOp>(loc, bounds->upperBound, remainder);
    auto vectorType = VectorType::get({vectorWidth}, rewriter.getI16Type());

    auto vectorLoop = rewriter.create<scf::ForOp>(
        loc, bounds->lowerBound, vectorEnd, vectorStep, ValueRange{adaptor.getInitial()},
        [&](OpBuilder &builder, Location bodyLoc, Value base, ValueRange iterArgs) {
          Value lhs = builder.create<vector::LoadOp>(bodyLoc, vectorType, adaptor.getLhs(), base);
          Value rhs = builder.create<vector::LoadOp>(bodyLoc, vectorType, adaptor.getRhs(), base);
          Value next = builder.create<ondrix::ondsp::ReduceMacOp>(
              bodyLoc, iterArgs.front().getType(), iterArgs.front(), lhs, rhs, numeric,
              *op.getProduct());
          builder.create<scf::YieldOp>(bodyLoc, next);
        });

    Value scalarStep = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    auto tailLoop = rewriter.create<scf::ForOp>(
        loc, vectorEnd, bounds->upperBound, scalarStep, ValueRange{vectorLoop.getResult(0)},
        [&](OpBuilder &builder, Location bodyLoc, Value index, ValueRange iterArgs) {
          Value lhs = builder.create<memref::LoadOp>(bodyLoc, adaptor.getLhs(), index);
          Value rhs = builder.create<memref::LoadOp>(bodyLoc, adaptor.getRhs(), index);
          Value next = builder.create<ondrix::ondsp::MacOp>(bodyLoc, iterArgs.front().getType(),
                                                            iterArgs.front(), lhs, rhs, numeric,
                                                            *op.getProduct());
          builder.create<scf::YieldOp>(bodyLoc, next);
        });

    rewriter.replaceOp(op, tailLoop.getResult(0));
    return success();
  }

private:
  int64_t vectorWidth;
};

class VectorizeOndspQ15MemRefReducePass final
    : public ondrix::impl::VectorizeOndspQ15MemRefReduceBase<VectorizeOndspQ15MemRefReducePass> {
public:
  using ondrix::impl::VectorizeOndspQ15MemRefReduceBase<
      VectorizeOndspQ15MemRefReducePass>::VectorizeOndspQ15MemRefReduceBase;

  void runOnOperation() override {
    if (vectorWidth <= 0) {
      getOperation().emitError("vector-width must be positive");
      signalPassFailure();
      return;
    }

    RewritePatternSet patterns(&getContext());
    patterns.add<ReduceMacOpVectorization>(&getContext(), vectorWidth);

    ConversionTarget target(getContext());
    target.addLegalDialect<arith::ArithDialect, cf::ControlFlowDialect, memref::MemRefDialect,
                           ondrix::ondsp::OndspDialect, scf::SCFDialect, vector::VectorDialect>();
    target.addDynamicallyLegalOp<ondrix::ondsp::ReduceMacOp>(
        [](ondrix::ondsp::ReduceMacOp op) { return !isSupportedMemRefReduction(op); });

    if (failed(applyPartialConversion(getOperation(), target, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> ondrix::createVectorizeOndspQ15MemRefReducePass() {
  return std::make_unique<VectorizeOndspQ15MemRefReducePass>();
}
