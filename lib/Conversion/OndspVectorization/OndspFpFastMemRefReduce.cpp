#include "ondrix/Conversion/OndspVectorization/OndspVectorization.h"
#include "ondrix/Conversion/Utils/MemRefLayoutUtils.h"
#include "ondrix/Conversion/Utils/ReductionUtils.h"

#include "ondrix/Dialect/ondsp/IR/OndspDialect.h"
#include "ondrix/Dialect/ondsp/IR/OndspOps.h"
#include "ondrix/Dialect/ondsp/IR/OndspSemantics.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace ondrix {
#define GEN_PASS_DEF_VECTORIZEONDSPFPFASTMEMREFREDUCE
#include "ondrix/Conversion/Passes.h.inc"
} // namespace ondrix

using namespace mlir;

namespace {

/// Largest accepted lane count, bounding every index the rewrite derives.
constexpr int64_t kMaxVectorWidth = 4096;

bool isSupportedFastMemRefReduction(ondrix::ondsp::ReduceMacOp op) {
  auto numeric = dyn_cast<ondrix::ondsp::FpAttr>(op.getNumeric());
  if (!numeric || !numeric.getFormat().isF32() ||
      numeric.getContract() != ondrix::ondsp::FpContractMode::Fast || op.getProduct() ||
      !op.getInitial().getType().isF32())
    return false;
  auto lhsType = dyn_cast<MemRefType>(op.getLhs().getType());
  auto rhsType = dyn_cast<MemRefType>(op.getRhs().getType());
  return lhsType && rhsType && lhsType.getRank() == 1 && rhsType.getRank() == 1 &&
         lhsType.getElementType().isF32() && rhsType.getElementType().isF32() &&
         ondrix::conversion::hasDefaultLLVMVectorMemorySpace(lhsType) &&
         ondrix::conversion::hasDefaultLLVMVectorMemorySpace(rhsType) &&
         isLastMemrefDimUnitStride(lhsType) && isLastMemrefDimUnitStride(rhsType);
}

class FastReduceMacOpVectorization final : public OpConversionPattern<ondrix::ondsp::ReduceMacOp> {
public:
  FastReduceMacOpVectorization(MLIRContext *context, int64_t vectorWidth)
      : OpConversionPattern(context), vectorWidth(vectorWidth) {}

  LogicalResult matchAndRewrite(ondrix::ondsp::ReduceMacOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    if (!isSupportedFastMemRefReduction(op))
      return failure();

    Type elementType = rewriter.getF32Type();
    FailureOr<ondrix::conversion::RankOneReductionBounds> bounds =
        ondrix::conversion::createRankOneMemRefReductionBounds(
            op, adaptor.getLhs(), adaptor.getRhs(), elementType, "fast f32 memref vectorization",
            rewriter);
    if (failed(bounds))
      return failure();

    Location loc = op.getLoc();
    auto fastFlags =
        arith::FastMathFlagsAttr::get(getContext(), ondrix::ondsp::getFastContractFlags());
    Value vectorStep = rewriter.create<arith::ConstantIndexOp>(loc, vectorWidth);
    Value remainder = rewriter.create<arith::RemUIOp>(loc, bounds->upperBound, vectorStep);
    Value vectorEnd = rewriter.create<arith::SubIOp>(loc, bounds->upperBound, remainder);
    auto vectorType = VectorType::get({vectorWidth}, elementType);
    Value zeroLanes = rewriter.create<arith::ConstantOp>(
        loc, vectorType, DenseElementsAttr::get(vectorType, rewriter.getF32FloatAttr(0.0f)));

    auto vectorLoop = rewriter.create<scf::ForOp>(
        loc, bounds->lowerBound, vectorEnd, vectorStep, ValueRange{zeroLanes},
        [&](OpBuilder &builder, Location bodyLoc, Value base, ValueRange iterArgs) {
          Value lhs = builder.create<vector::LoadOp>(bodyLoc, vectorType, adaptor.getLhs(), base);
          Value rhs = builder.create<vector::LoadOp>(bodyLoc, vectorType, adaptor.getRhs(), base);
          Value next = builder.create<math::FmaOp>(bodyLoc, lhs, rhs, iterArgs.front(), fastFlags);
          builder.create<scf::YieldOp>(bodyLoc, next);
        });

    // MLIR 17's vector.reduction carries no fastmath attribute; it lowers to
    // the ordered llvm.intr.vector.reduce.fadd, a fixed deterministic
    // regrouping the declared relaxation authorizes.
    Value reduced = rewriter.create<vector::ReductionOp>(loc, vector::CombiningKind::ADD,
                                                         vectorLoop.getResult(0));

    Value scalarStep = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    auto tailLoop = rewriter.create<scf::ForOp>(
        loc, vectorEnd, bounds->upperBound, scalarStep, ValueRange{reduced},
        [&](OpBuilder &builder, Location bodyLoc, Value index, ValueRange iterArgs) {
          Value lhs = builder.create<memref::LoadOp>(bodyLoc, adaptor.getLhs(), index);
          Value rhs = builder.create<memref::LoadOp>(bodyLoc, adaptor.getRhs(), index);
          Value next = builder.create<math::FmaOp>(bodyLoc, lhs, rhs, iterArgs.front(), fastFlags);
          builder.create<scf::YieldOp>(bodyLoc, next);
        });

    rewriter.replaceOpWithNewOp<arith::AddFOp>(op, tailLoop.getResult(0), adaptor.getInitial(),
                                               fastFlags);
    return success();
  }

private:
  int64_t vectorWidth;
};

class VectorizeOndspFpFastMemRefReducePass final
    : public ondrix::impl::VectorizeOndspFpFastMemRefReduceBase<
          VectorizeOndspFpFastMemRefReducePass> {
public:
  using ondrix::impl::VectorizeOndspFpFastMemRefReduceBase<
      VectorizeOndspFpFastMemRefReducePass>::VectorizeOndspFpFastMemRefReduceBase;

  void runOnOperation() override {
    if (vectorWidth <= 1) {
      getOperation().emitError("vector-width must be greater than one");
      signalPassFailure();
      return;
    }
    if (vectorWidth > kMaxVectorWidth) {
      getOperation().emitError("vector-width must not exceed ") << kMaxVectorWidth;
      signalPassFailure();
      return;
    }

    RewritePatternSet patterns(&getContext());
    patterns.add<FastReduceMacOpVectorization>(&getContext(), vectorWidth);

    ConversionTarget target(getContext());
    target.addLegalDialect<arith::ArithDialect, cf::ControlFlowDialect, math::MathDialect,
                           memref::MemRefDialect, ondrix::ondsp::OndspDialect, scf::SCFDialect,
                           vector::VectorDialect>();
    target.addDynamicallyLegalOp<ondrix::ondsp::ReduceMacOp>(
        [](ondrix::ondsp::ReduceMacOp op) { return !isSupportedFastMemRefReduction(op); });

    if (failed(applyPartialConversion(getOperation(), target, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> ondrix::createVectorizeOndspFpFastMemRefReducePass() {
  return std::make_unique<VectorizeOndspFpFastMemRefReducePass>();
}

std::unique_ptr<Pass> ondrix::createVectorizeOndspFpFastMemRefReducePass(
    const ondrix::VectorizeOndspFpFastMemRefReduceOptions &options) {
  return std::make_unique<VectorizeOndspFpFastMemRefReducePass>(options);
}
