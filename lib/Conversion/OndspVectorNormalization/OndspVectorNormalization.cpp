#include "ondrix/Conversion/OndspVectorNormalization/OndspVectorNormalization.h"
#include "ondrix/Conversion/Utils/FixedPointDomainUtils.h"

#include "ondrix/Dialect/ondsp/IR/OndspDialect.h"
#include "ondrix/Dialect/ondsp/IR/OndspOps.h"
#include "ondrix/Dialect/ondsp/IR/OndspSemantics.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace ondrix {
#define GEN_PASS_DEF_NORMALIZEONDSPFIXEDVECTORREDUCE
#define GEN_PASS_DEF_PARALLELIZEONDSPFIXEDWRAPVECTORREDUCE
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
         lhsType.getElementType() == numeric.getStorage() &&
         ondrix::conversion::isSupportedFixedVectorMacDomain(accumulator, numeric,
                                                             *op.getProduct());
}

static bool isParallelizableWrapReduction(ondrix::ondsp::ReduceMacOp op) {
  if (!isSupportedVectorReduction(op))
    return false;
  auto accumulator = cast<ondrix::ondsp::AccType>(op.getInitial().getType());
  return ondrix::ondsp::classifyReductionReassociation(accumulator.getUpdateOverflow()) ==
         ondrix::ondsp::ReductionReassociationSafety::ExactModulo;
}

struct LoweredVectorTerms {
  Value terms;
  ondrix::ondsp::FixedAttr numeric;
};

static FailureOr<LoweredVectorTerms> lowerVectorTerms(ondrix::ondsp::ReduceMacOp op, Value lhs,
                                                      Value rhs,
                                                      ConversionPatternRewriter &rewriter) {
  auto accumulator = cast<ondrix::ondsp::AccType>(op.getInitial().getType());
  auto numeric = cast<ondrix::ondsp::FixedAttr>(op.getNumeric());
  FailureOr<ondrix::conversion::SupportedFixedMacDomain> domain =
      ondrix::conversion::getSupportedFixedVectorMacDomain(op, accumulator, numeric,
                                                           *op.getProduct());
  if (failed(domain))
    return failure();

  auto vectorType = cast<VectorType>(op.getLhs().getType());
  auto fullProductVectorType = VectorType::get(vectorType.getShape(), domain->fullProductStorage);
  Value extendedLhs = rewriter.create<arith::ExtSIOp>(op.getLoc(), fullProductVectorType, lhs);
  Value extendedRhs = rewriter.create<arith::ExtSIOp>(op.getLoc(), fullProductVectorType, rhs);
  Value fullProducts = rewriter.create<arith::MulIOp>(op.getLoc(), extendedLhs, extendedRhs);

  Value terms = fullProducts;
  if (domain->product.selection == ondrix::ondsp::ProductSelection::HighRaw) {
    IntegerAttr shiftValue =
        rewriter.getIntegerAttr(domain->fullProductStorage, domain->operandStorage.getWidth());
    auto shiftValues = SplatElementsAttr::get(fullProductVectorType, shiftValue);
    Value shift =
        rewriter.create<arith::ConstantOp>(op.getLoc(), fullProductVectorType, shiftValues);
    Value shifted = rewriter.create<arith::ShRSIOp>(op.getLoc(), fullProducts, shift);
    auto termVectorType = VectorType::get(vectorType.getShape(), domain->termStorage);
    terms = rewriter.create<arith::TruncIOp>(op.getLoc(), termVectorType, shifted);
  }

  auto termNumeric = ondrix::ondsp::FixedAttr::get(rewriter.getContext(), domain->signedness,
                                                   domain->termStorage, domain->product.frac);
  return LoweredVectorTerms{terms, termNumeric};
}

class WrapReduceMacOpParallelization final
    : public OpConversionPattern<ondrix::ondsp::ReduceMacOp> {
public:
  using OpConversionPattern<ondrix::ondsp::ReduceMacOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(ondrix::ondsp::ReduceMacOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    if (!isParallelizableWrapReduction(op))
      return failure();

    FailureOr<LoweredVectorTerms> lowered =
        lowerVectorTerms(op, adaptor.getLhs(), adaptor.getRhs(), rewriter);
    if (failed(lowered))
      return failure();

    auto termVectorType = cast<VectorType>(lowered->terms.getType());
    auto wideVectorType = VectorType::get(termVectorType.getShape(), rewriter.getI64Type());
    Value wideTerms = lowered->terms;
    if (termVectorType.getElementType() != rewriter.getI64Type())
      wideTerms = rewriter.create<arith::ExtSIOp>(op.getLoc(), wideVectorType, lowered->terms);
    // Modular i64 addition preserves every low bit of the supported i40/i64 accumulators.
    Value partialSum =
        rewriter.create<vector::ReductionOp>(op.getLoc(), vector::CombiningKind::ADD, wideTerms);
    auto partialNumeric =
        ondrix::ondsp::FixedAttr::get(rewriter.getContext(), lowered->numeric.getSignedness(),
                                      rewriter.getI64Type(), lowered->numeric.getFrac());

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

    FailureOr<LoweredVectorTerms> lowered =
        lowerVectorTerms(op, adaptor.getLhs(), adaptor.getRhs(), rewriter);
    if (failed(lowered))
      return failure();

    auto vectorType = cast<VectorType>(lowered->terms.getType());
    Value accumulator = adaptor.getInitial();
    // Preserve the universal left-fold order; saturating updates are not generally associative.
    for (int64_t lane = 0; lane < vectorType.getNumElements(); ++lane) {
      Value term = rewriter.create<vector::ExtractOp>(op.getLoc(), lowered->terms, lane);
      accumulator = rewriter.create<ondrix::ondsp::AccAddTermOp>(
          op.getLoc(), accumulator.getType(), accumulator, term, lowered->numeric);
    }

    rewriter.replaceOp(op, accumulator);
    return success();
  }
};

class NormalizeOndspFixedVectorReducePass final
    : public ondrix::impl::NormalizeOndspFixedVectorReduceBase<
          NormalizeOndspFixedVectorReducePass> {
public:
  using ondrix::impl::NormalizeOndspFixedVectorReduceBase<
      NormalizeOndspFixedVectorReducePass>::NormalizeOndspFixedVectorReduceBase;

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

class ParallelizeOndspFixedWrapVectorReducePass final
    : public ondrix::impl::ParallelizeOndspFixedWrapVectorReduceBase<
          ParallelizeOndspFixedWrapVectorReducePass> {
public:
  using ondrix::impl::ParallelizeOndspFixedWrapVectorReduceBase<
      ParallelizeOndspFixedWrapVectorReducePass>::ParallelizeOndspFixedWrapVectorReduceBase;

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

std::unique_ptr<Pass> ondrix::createNormalizeOndspFixedVectorReducePass() {
  return std::make_unique<NormalizeOndspFixedVectorReducePass>();
}

std::unique_ptr<Pass> ondrix::createParallelizeOndspFixedWrapVectorReducePass() {
  return std::make_unique<ParallelizeOndspFixedWrapVectorReducePass>();
}
