#include "ondrix/Conversion/OndspToScalar/OndspToScalar.h"
#include "ondrix/Conversion/Utils/ReductionUtils.h"
#include "ondrix/Dialect/ondsp/IR/OndspSemantics.h"

#include "ondrix/Dialect/ondsp/IR/OndspAttrs.h"
#include "ondrix/Dialect/ondsp/IR/OndspDialect.h"
#include "ondrix/Dialect/ondsp/IR/OndspOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace ondrix {
#define GEN_PASS_DEF_LOWERONDSPF32REDUCETOSCALAR
#include "ondrix/Conversion/Passes.h.inc"
} // namespace ondrix

using namespace mlir;

namespace {

constexpr int64_t kMaxOrderedProductBlock = 4096;

static bool isSupportedF32MemRefReduction(ondrix::ondsp::ReduceMacOp op) {
  auto numeric = dyn_cast<ondrix::ondsp::FpAttr>(op.getNumeric());
  auto lhsType = dyn_cast<MemRefType>(op.getLhs().getType());
  auto rhsType = dyn_cast<MemRefType>(op.getRhs().getType());
  return numeric && numeric.getFormat().isF32() && lhsType && rhsType && lhsType.getRank() == 1 &&
         rhsType.getRank() == 1 && lhsType.getElementType().isF32() &&
         rhsType.getElementType().isF32() && op.getInitial().getType().isF32() &&
         op.getResult().getType().isF32();
}

static bool isSupportedF32ComplexMemRefReduction(ondrix::ondsp::CxReduceMacOp op) {
  auto numeric = dyn_cast<ondrix::ondsp::FpAttr>(op.getNumeric());
  return numeric && numeric.getFormat().isF32() && isa<MemRefType>(op.getLhs().getType());
}

// One complex term under the declared contract: `a*b + c*d`, or `a*b - c*d`
// when `subtract`. The sum seeds on the first product, so the fused modes
// spend one update where `off` spends a multiply and an add.
static Value createFpComplexTerm(Location loc, Value a, Value b, Value c, Value d, bool subtract,
                                 ondrix::ondsp::FpAttr numeric, OpBuilder &builder) {
  Value seed = builder.create<arith::MulFOp>(loc, a, b);
  if (numeric.getContract() == ondrix::ondsp::FpContractMode::Off) {
    Value product = builder.create<arith::MulFOp>(loc, c, d);
    return subtract ? builder.create<arith::SubFOp>(loc, seed, product).getResult()
                    : builder.create<arith::AddFOp>(loc, seed, product).getResult();
  }
  // Negating a multiplicand is exact, so the subtraction rides the same fused
  // update rather than costing a rounding of its own.
  Value multiplicand = subtract ? builder.create<arith::NegFOp>(loc, c).getResult() : c;
  Value fused = builder.create<math::FmaOp>(loc, multiplicand, d, seed);
  if (numeric.getContract() == ondrix::ondsp::FpContractMode::Fast)
    return ondrix::ondsp::consumeFastPermission(fused.getDefiningOp(),
                                                ondrix::ondsp::FastPermission::FuseMultiplyAdd);
  return fused;
}

// The interleaved f32 complex reduction: the same pairing the packed profile
// declares, over two adjacent elements per value and with the format itself
// as the carrier, so nothing here requantizes.
class CxReduceMacOpLowering final : public OpConversionPattern<ondrix::ondsp::CxReduceMacOp> {
public:
  using OpConversionPattern<ondrix::ondsp::CxReduceMacOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(ondrix::ondsp::CxReduceMacOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto numeric = dyn_cast<ondrix::ondsp::FpAttr>(op.getNumeric());
    if (!numeric || !numeric.getFormat().isF32())
      return rewriter.notifyMatchFailure(op, "not the interleaved f32 complex profile");
    FailureOr<ondrix::conversion::RankOneReductionBounds> bounds =
        ondrix::conversion::createRankOneMemRefReductionBounds(
            op, adaptor.getLhs(), adaptor.getRhs(), rewriter.getF32Type(),
            "interleaved f32 complex scalar lowering", rewriter);
    if (failed(bounds))
      return failure();

    Location loc = op.getLoc();
    bool conjugate = op.getConjugate();
    Value two = rewriter.create<arith::ConstantIndexOp>(loc, 2);
    Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    // The bounds walk elements; a complex value is two of them.
    Value values = rewriter.create<arith::DivUIOp>(loc, bounds->upperBound, two);
    auto loop = rewriter.create<scf::ForOp>(
        loc, bounds->lowerBound, values, one,
        ValueRange{adaptor.getInitialReal(), adaptor.getInitialImag()},
        [&](OpBuilder &builder, Location bodyLoc, Value index, ValueRange iterArgs) {
          Value realIndex = builder.create<arith::MulIOp>(bodyLoc, index, two);
          Value imagIndex = builder.create<arith::AddIOp>(bodyLoc, realIndex, one);
          Value xr = builder.create<memref::LoadOp>(bodyLoc, adaptor.getLhs(), realIndex);
          Value xi = builder.create<memref::LoadOp>(bodyLoc, adaptor.getLhs(), imagIndex);
          Value yr = builder.create<memref::LoadOp>(bodyLoc, adaptor.getRhs(), realIndex);
          Value yi = builder.create<memref::LoadOp>(bodyLoc, adaptor.getRhs(), imagIndex);
          Value realTerm =
              createFpComplexTerm(bodyLoc, xr, yr, xi, yi, !conjugate, numeric, builder);
          Value imagTerm = conjugate ? createFpComplexTerm(bodyLoc, xi, yr, xr, yi,
                                                           /*subtract=*/true, numeric, builder)
                                     : createFpComplexTerm(bodyLoc, xr, yi, xi, yr,
                                                           /*subtract=*/false, numeric, builder);
          Value nextReal = builder.create<arith::AddFOp>(bodyLoc, iterArgs[0], realTerm);
          Value nextImag = builder.create<arith::AddFOp>(bodyLoc, iterArgs[1], imagTerm);
          builder.create<scf::YieldOp>(bodyLoc, ValueRange{nextReal, nextImag});
        });
    rewriter.replaceOp(op, loop.getResults());
    return success();
  }
};

class ReduceMacOpLowering final : public OpConversionPattern<ondrix::ondsp::ReduceMacOp> {
public:
  ReduceMacOpLowering(MLIRContext *context, int64_t vectorWidth)
      : OpConversionPattern(context), vectorWidth(vectorWidth) {}

  LogicalResult matchAndRewrite(ondrix::ondsp::ReduceMacOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto numeric = dyn_cast<ondrix::ondsp::FpAttr>(op.getNumeric());
    if (!numeric || !numeric.getFormat().isF32())
      return op.emitOpError("scalar lowering requires numeric = #ondsp.fp<format = f32, ...>");
    if (op.getProduct())
      return op.emitOpError(
          "scalar floating-point reduce_mac lowering requires no product attribute");
    if (!op.getInitial().getType().isF32() || !op.getResult().getType().isF32())
      return op.emitOpError("scalar lowering requires an f32 initial value and result");

    FailureOr<ondrix::conversion::RankOneReductionBounds> bounds =
        ondrix::conversion::createRankOneMemRefReductionBounds(
            op, adaptor.getLhs(), adaptor.getRhs(), rewriter.getF32Type(), "f32 scalar lowering",
            rewriter);
    if (failed(bounds))
      return failure();

    Location loc = op.getLoc();
    // Under a declared-off contract each product rounds once, as its scalar
    // multiply would, and indices are independent, so a block of products is
    // a rescheduling; only the folds must stay in index order.
    Value seed = adaptor.getInitial();
    Value scalarStart = bounds->lowerBound;
    auto lhsType = dyn_cast<MemRefType>(adaptor.getLhs().getType());
    auto rhsType = dyn_cast<MemRefType>(adaptor.getRhs().getType());
    bool contiguous = lhsType && rhsType && isLastMemrefDimUnitStride(lhsType) &&
                      isLastMemrefDimUnitStride(rhsType);
    if (vectorWidth > 1 && contiguous &&
        numeric.getContract() == ondrix::ondsp::FpContractMode::Off) {
      Value blockStep = rewriter.create<arith::ConstantIndexOp>(loc, vectorWidth);
      Value remainder = rewriter.create<arith::RemUIOp>(loc, bounds->upperBound, blockStep);
      Value blockEnd = rewriter.create<arith::SubIOp>(loc, bounds->upperBound, remainder);
      auto vectorType = VectorType::get({vectorWidth}, rewriter.getF32Type());
      auto blockLoop = rewriter.create<scf::ForOp>(
          loc, bounds->lowerBound, blockEnd, blockStep, ValueRange{seed},
          [&](OpBuilder &builder, Location bodyLoc, Value base, ValueRange iterArgs) {
            Value lhs = builder.create<vector::LoadOp>(bodyLoc, vectorType, adaptor.getLhs(), base);
            Value rhs = builder.create<vector::LoadOp>(bodyLoc, vectorType, adaptor.getRhs(), base);
            Value products = builder.create<arith::MulFOp>(bodyLoc, lhs, rhs);
            Value accumulator = iterArgs.front();
            for (int64_t lane = 0; lane < vectorWidth; ++lane) {
              Value product = builder.create<vector::ExtractOp>(bodyLoc, products, lane);
              accumulator = builder.create<arith::AddFOp>(bodyLoc, accumulator, product);
            }
            builder.create<scf::YieldOp>(bodyLoc, accumulator);
          });
      seed = blockLoop.getResult(0);
      scalarStart = blockEnd;
    }
    Value step = rewriter.create<arith::ConstantIndexOp>(loc, 1);

    auto loop = rewriter.create<scf::ForOp>(
        loc, scalarStart, bounds->upperBound, step, ValueRange{seed},
        [&](OpBuilder &builder, Location bodyLoc, Value iv, ValueRange iterArgs) {
          Value lhs = builder.create<memref::LoadOp>(bodyLoc, adaptor.getLhs(), iv);
          Value rhs = builder.create<memref::LoadOp>(bodyLoc, adaptor.getRhs(), iv);
          Value next;
          switch (numeric.getContract()) {
          case ondrix::ondsp::FpContractMode::Fma:
            next = builder.create<math::FmaOp>(bodyLoc, lhs, rhs, iterArgs.front());
            break;
          case ondrix::ondsp::FpContractMode::Off: {
            Value product = builder.create<arith::MulFOp>(bodyLoc, lhs, rhs);
            next = builder.create<arith::AddFOp>(bodyLoc, iterArgs.front(), product);
            break;
          }
          case ondrix::ondsp::FpContractMode::Fast:
            // Ordered scalar route: R goes unused, F is spent on the fused
            // chain.
            next = ondrix::ondsp::consumeFastPermission(
                builder.create<math::FmaOp>(bodyLoc, lhs, rhs, iterArgs.front()),
                ondrix::ondsp::FastPermission::FuseMultiplyAdd);
            break;
          }
          builder.create<scf::YieldOp>(bodyLoc, next);
        });

    rewriter.replaceOp(op, loop.getResult(0));
    return success();
  }

private:
  int64_t vectorWidth;
};

class LowerOndspF32ReduceToScalarPass final
    : public ondrix::impl::LowerOndspF32ReduceToScalarBase<LowerOndspF32ReduceToScalarPass> {
public:
  using ondrix::impl::LowerOndspF32ReduceToScalarBase<
      LowerOndspF32ReduceToScalarPass>::LowerOndspF32ReduceToScalarBase;

  void runOnOperation() override {
    // The block loop emits one fold per lane, so an unbounded width is a
    // compile-time expansion, not a schedule.
    if (vectorWidth < 1 || vectorWidth > kMaxOrderedProductBlock) {
      getOperation().emitError("vector-width must be between 1 and ") << kMaxOrderedProductBlock;
      signalPassFailure();
      return;
    }
    RewritePatternSet patterns(&getContext());
    patterns.add<ReduceMacOpLowering>(&getContext(), vectorWidth);
    patterns.add<CxReduceMacOpLowering>(&getContext());

    ConversionTarget target(getContext());
    target.addLegalDialect<BuiltinDialect, arith::ArithDialect, cf::ControlFlowDialect,
                           func::FuncDialect, math::MathDialect, memref::MemRefDialect,
                           scf::SCFDialect, vector::VectorDialect, ondrix::ondsp::OndspDialect>();
    target.addDynamicallyLegalOp<ondrix::ondsp::ReduceMacOp>(
        [](ondrix::ondsp::ReduceMacOp op) { return !isSupportedF32MemRefReduction(op); });
    target.addDynamicallyLegalOp<ondrix::ondsp::CxReduceMacOp>(
        [](ondrix::ondsp::CxReduceMacOp op) { return !isSupportedF32ComplexMemRefReduction(op); });

    if (failed(applyPartialConversion(getOperation(), target, std::move(patterns))))
      return signalPassFailure();
    // The declared-contract stamp's only audience is the schedule stage,
    // which has run by now; a residual stamp on an unbatched loop would trip
    // the host-metadata artifact guard downstream.
    getOperation()->walk(
        [](Operation *op) { op->removeAttr(ondrix::ondsp::getDeclaredNumericAttrName()); });
    ondrix::ondsp::summarizeFastPermissions(getOperation());
  }
};

} // namespace

std::unique_ptr<Pass> ondrix::createLowerOndspF32ReduceToScalarPass() {
  return std::make_unique<LowerOndspF32ReduceToScalarPass>();
}
