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

static bool isSupportedF32MemRefReduction(ondrix::ondsp::ReduceMacOp op) {
  auto numeric = dyn_cast<ondrix::ondsp::FpAttr>(op.getNumeric());
  auto lhsType = dyn_cast<MemRefType>(op.getLhs().getType());
  auto rhsType = dyn_cast<MemRefType>(op.getRhs().getType());
  return numeric && numeric.getFormat().isF32() && lhsType && rhsType && lhsType.getRank() == 1 &&
         rhsType.getRank() == 1 && lhsType.getElementType().isF32() &&
         rhsType.getElementType().isF32() && op.getInitial().getType().isF32() &&
         op.getResult().getType().isF32();
}

class ReduceMacOpLowering final : public OpConversionPattern<ondrix::ondsp::ReduceMacOp> {
public:
  using OpConversionPattern<ondrix::ondsp::ReduceMacOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(ondrix::ondsp::ReduceMacOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto numeric = dyn_cast<ondrix::ondsp::FpAttr>(op.getNumeric());
    if (!numeric || !numeric.getFormat().isF32()) {
      op.emitOpError("scalar lowering requires numeric = #ondsp.fp<format = f32, ...>");
      return failure();
    }
    if (op.getProduct()) {
      op.emitOpError("scalar floating-point reduce_mac lowering requires no product attribute");
      return failure();
    }
    if (!op.getInitial().getType().isF32() || !op.getResult().getType().isF32()) {
      op.emitOpError("scalar lowering requires an f32 initial value and result");
      return failure();
    }

    FailureOr<ondrix::conversion::RankOneReductionBounds> bounds =
        ondrix::conversion::createRankOneMemRefReductionBounds(
            op, adaptor.getLhs(), adaptor.getRhs(), rewriter.getF32Type(), "f32 scalar lowering",
            rewriter);
    if (failed(bounds))
      return failure();

    Location loc = op.getLoc();
    Value step = rewriter.create<arith::ConstantIndexOp>(loc, 1);

    auto loop = rewriter.create<scf::ForOp>(
        loc, bounds->lowerBound, bounds->upperBound, step, ValueRange{adaptor.getInitial()},
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
};

class LowerOndspF32ReduceToScalarPass final
    : public ondrix::impl::LowerOndspF32ReduceToScalarBase<LowerOndspF32ReduceToScalarPass> {
public:
  using ondrix::impl::LowerOndspF32ReduceToScalarBase<
      LowerOndspF32ReduceToScalarPass>::LowerOndspF32ReduceToScalarBase;

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<ReduceMacOpLowering>(&getContext());

    ConversionTarget target(getContext());
    target.addLegalDialect<BuiltinDialect, arith::ArithDialect, cf::ControlFlowDialect,
                           func::FuncDialect, math::MathDialect, memref::MemRefDialect,
                           scf::SCFDialect, ondrix::ondsp::OndspDialect>();
    target.addDynamicallyLegalOp<ondrix::ondsp::ReduceMacOp>(
        [](ondrix::ondsp::ReduceMacOp op) { return !isSupportedF32MemRefReduction(op); });

    if (failed(applyPartialConversion(getOperation(), target, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> ondrix::createLowerOndspF32ReduceToScalarPass() {
  return std::make_unique<LowerOndspF32ReduceToScalarPass>();
}
