#include "ondrix/Conversion/OndspVectorNormalization/OndspVectorNormalization.h"

#include "ondrix/Dialect/ondsp/IR/OndspDialect.h"
#include "ondrix/Dialect/ondsp/IR/OndspOps.h"
#include "ondrix/Dialect/ondsp/IR/OndspSemantics.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace ondrix {
#define GEN_PASS_DEF_NORMALIZEONDSPQ15VECTORREDUCE
#define GEN_PASS_DEF_PARALLELIZEONDSPQ15WRAPVECTORREDUCE
#include "ondrix/Conversion/Passes.h.inc"
} // namespace ondrix

using namespace mlir;

namespace {

static bool isSupportedVectorReduction(ondrix::ondsp::ReduceMacOp op) {
  auto accumulator = dyn_cast<ondrix::ondsp::AccType>(op.getInitial().getType());
  auto numeric = dyn_cast<ondrix::ondsp::FixedAttr>(op.getNumeric());
  auto lhsType = dyn_cast<VectorType>(op.getLhs().getType());
  auto rhsType = dyn_cast<VectorType>(op.getRhs().getType());
  return accumulator && numeric && op.getProduct() && lhsType && rhsType && !lhsType.isScalable() &&
         !rhsType.isScalable() && lhsType.getRank() == 1 && lhsType == rhsType &&
         lhsType.getElementType().isSignlessInteger(16) &&
         ondrix::ondsp::isSignedQ15I40Accumulator(accumulator) &&
         ondrix::ondsp::isSignedQ15(numeric) && ondrix::ondsp::isFullProduct(*op.getProduct());
}

static bool isParallelizableWrapReduction(ondrix::ondsp::ReduceMacOp op) {
  if (!isSupportedVectorReduction(op))
    return false;
  auto accumulator = cast<ondrix::ondsp::AccType>(op.getInitial().getType());
  return ondrix::ondsp::classifyReductionReassociation(accumulator.getUpdateOverflow()) ==
         ondrix::ondsp::ReductionReassociationSafety::ExactModulo;
}

class WrapReduceMacOpParallelization final
    : public OpConversionPattern<ondrix::ondsp::ReduceMacOp> {
public:
  using OpConversionPattern<ondrix::ondsp::ReduceMacOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(ondrix::ondsp::ReduceMacOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    if (!isParallelizableWrapReduction(op))
      return failure();

    auto vectorType = cast<VectorType>(op.getLhs().getType());
    auto productVectorType = VectorType::get(vectorType.getShape(), rewriter.getI32Type());
    Value lhs = rewriter.create<arith::ExtSIOp>(op.getLoc(), productVectorType, adaptor.getLhs());
    Value rhs = rewriter.create<arith::ExtSIOp>(op.getLoc(), productVectorType, adaptor.getRhs());
    Value products = rewriter.create<arith::MulIOp>(op.getLoc(), lhs, rhs);

    auto wideVectorType = VectorType::get(vectorType.getShape(), rewriter.getI64Type());
    Value wideProducts = rewriter.create<arith::ExtSIOp>(op.getLoc(), wideVectorType, products);
    // An i64 modular sum preserves the low 40 bits required by the wrapping accumulator.
    Value partialSum =
        rewriter.create<vector::ReductionOp>(op.getLoc(), vector::CombiningKind::ADD, wideProducts);
    auto partialNumeric = ondrix::ondsp::FixedAttr::get(
        rewriter.getContext(), ondrix::ondsp::Signedness::Signed, rewriter.getI64Type(), 30);

    rewriter.replaceOpWithNewOp<ondrix::ondsp::AccAddTermOp>(
        op, op.getResult().getType(), adaptor.getInitial(), partialSum, partialNumeric);
    return success();
  }
};

class ReduceMacOpLowering final : public OpConversionPattern<ondrix::ondsp::ReduceMacOp> {
public:
  using OpConversionPattern<ondrix::ondsp::ReduceMacOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(ondrix::ondsp::ReduceMacOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    if (!isSupportedVectorReduction(op))
      return failure();

    auto vectorType = cast<VectorType>(op.getLhs().getType());
    auto productVectorType = VectorType::get(vectorType.getShape(), rewriter.getI32Type());
    Value lhs = rewriter.create<arith::ExtSIOp>(op.getLoc(), productVectorType, adaptor.getLhs());
    Value rhs = rewriter.create<arith::ExtSIOp>(op.getLoc(), productVectorType, adaptor.getRhs());
    Value products = rewriter.create<arith::MulIOp>(op.getLoc(), lhs, rhs);
    auto productNumeric = ondrix::ondsp::FixedAttr::get(
        rewriter.getContext(), ondrix::ondsp::Signedness::Signed, rewriter.getI32Type(), 30);

    Value accumulator = adaptor.getInitial();
    // Preserve the universal left-fold order; saturating updates are not generally associative.
    for (int64_t lane = 0; lane < vectorType.getNumElements(); ++lane) {
      Value product = rewriter.create<vector::ExtractOp>(op.getLoc(), products, lane);
      accumulator = rewriter.create<ondrix::ondsp::AccAddTermOp>(
          op.getLoc(), accumulator.getType(), accumulator, product, productNumeric);
    }

    rewriter.replaceOp(op, accumulator);
    return success();
  }
};

class NormalizeOndspQ15VectorReducePass final
    : public ondrix::impl::NormalizeOndspQ15VectorReduceBase<NormalizeOndspQ15VectorReducePass> {
public:
  using ondrix::impl::NormalizeOndspQ15VectorReduceBase<
      NormalizeOndspQ15VectorReducePass>::NormalizeOndspQ15VectorReduceBase;

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<ReduceMacOpLowering>(&getContext());

    ConversionTarget target(getContext());
    target
        .addLegalDialect<arith::ArithDialect, ondrix::ondsp::OndspDialect, vector::VectorDialect>();
    target.addDynamicallyLegalOp<ondrix::ondsp::ReduceMacOp>(
        [](ondrix::ondsp::ReduceMacOp op) { return !isSupportedVectorReduction(op); });

    if (failed(applyPartialConversion(getOperation(), target, std::move(patterns))))
      signalPassFailure();
  }
};

class ParallelizeOndspQ15WrapVectorReducePass final
    : public ondrix::impl::ParallelizeOndspQ15WrapVectorReduceBase<
          ParallelizeOndspQ15WrapVectorReducePass> {
public:
  using ondrix::impl::ParallelizeOndspQ15WrapVectorReduceBase<
      ParallelizeOndspQ15WrapVectorReducePass>::ParallelizeOndspQ15WrapVectorReduceBase;

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<WrapReduceMacOpParallelization>(&getContext());

    ConversionTarget target(getContext());
    target
        .addLegalDialect<arith::ArithDialect, ondrix::ondsp::OndspDialect, vector::VectorDialect>();
    target.addDynamicallyLegalOp<ondrix::ondsp::ReduceMacOp>(
        [](ondrix::ondsp::ReduceMacOp op) { return !isParallelizableWrapReduction(op); });

    if (failed(applyPartialConversion(getOperation(), target, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> ondrix::createNormalizeOndspQ15VectorReducePass() {
  return std::make_unique<NormalizeOndspQ15VectorReducePass>();
}

std::unique_ptr<Pass> ondrix::createParallelizeOndspQ15WrapVectorReducePass() {
  return std::make_unique<ParallelizeOndspQ15WrapVectorReducePass>();
}
