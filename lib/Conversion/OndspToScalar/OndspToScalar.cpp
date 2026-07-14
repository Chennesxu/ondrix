#include "ondrix/Conversion/OndspToScalar/OndspToScalar.h"

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
#define GEN_PASS_DEF_CONVERTONDSPTOSCALAR
#include "ondrix/Conversion/Passes.h.inc"
} // namespace ondrix

using namespace mlir;

namespace {

static FailureOr<MemRefType> getRankOneF32MemRefType(Operation *op, Value value,
                                                     StringRef operandName) {
  auto type = value.getType().dyn_cast<MemRefType>();
  if (!type || type.getRank() != 1)
    return op->emitOpError() << operandName
                             << " must be a rank-1 memref<Nxf32> for scalar lowering";
  if (!type.getElementType().isF32())
    return op->emitOpError() << operandName << " must have f32 elements for scalar lowering";
  return type;
}

static Value getDimZeroSize(Location loc, Value value, MemRefType type, Value zeroIndex,
                            ConversionPatternRewriter &rewriter) {
  if (!type.isDynamicDim(0))
    return rewriter.create<arith::ConstantIndexOp>(loc, type.getDimSize(0));
  return rewriter.create<memref::DimOp>(loc, value, zeroIndex);
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
    if (!op.getResult().getType().isF32()) {
      op.emitOpError("scalar lowering requires an f32 result");
      return failure();
    }

    FailureOr<MemRefType> lhsMemRefType = getRankOneF32MemRefType(op, op.getLhs(), "lhs");
    if (failed(lhsMemRefType))
      return failure();
    FailureOr<MemRefType> rhsMemRefType = getRankOneF32MemRefType(op, op.getRhs(), "rhs");
    if (failed(rhsMemRefType))
      return failure();

    if (!lhsMemRefType->isDynamicDim(0) && !rhsMemRefType->isDynamicDim(0) &&
        lhsMemRefType->getDimSize(0) != rhsMemRefType->getDimSize(0)) {
      op.emitOpError("scalar lowering requires lhs and rhs to have equal static lengths");
      return failure();
    }

    Location loc = op.getLoc();
    Value zero = rewriter.create<arith::ConstantOp>(loc, rewriter.getF32FloatAttr(0.0));
    Value lowerBound = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value upperBound = getDimZeroSize(loc, adaptor.getLhs(), *lhsMemRefType, lowerBound, rewriter);
    if (lhsMemRefType->isDynamicDim(0) || rhsMemRefType->isDynamicDim(0)) {
      Value rhsSize = getDimZeroSize(loc, adaptor.getRhs(), *rhsMemRefType, lowerBound, rewriter);
      // The scalar correctness path uses the lhs extent as its loop bound, so
      // dynamic shapes require a runtime proof that both extents agree.
      Value lengthsMatch =
          rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, upperBound, rhsSize);
      rewriter.create<cf::AssertOp>(
          loc, lengthsMatch,
          rewriter.getStringAttr("ondsp.reduce_mac requires equal operand lengths"));
    }
    Value step = rewriter.create<arith::ConstantIndexOp>(loc, 1);

    auto loop = rewriter.create<scf::ForOp>(
        loc, lowerBound, upperBound, step, ValueRange{zero},
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
            next = builder.create<math::FmaOp>(bodyLoc, lhs, rhs, iterArgs.front(),
                                               arith::FastMathFlags::fast);
            break;
          }
          builder.create<scf::YieldOp>(bodyLoc, next);
        });

    rewriter.replaceOp(op, loop.getResult(0));
    return success();
  }
};

class ConvertOndspToScalarPass final
    : public ondrix::impl::ConvertOndspToScalarBase<ConvertOndspToScalarPass> {
public:
  using ondrix::impl::ConvertOndspToScalarBase<ConvertOndspToScalarPass>::ConvertOndspToScalarBase;

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<ReduceMacOpLowering>(&getContext());

    ConversionTarget target(getContext());
    target.addLegalDialect<BuiltinDialect, arith::ArithDialect, cf::ControlFlowDialect,
                           func::FuncDialect, math::MathDialect, memref::MemRefDialect,
                           scf::SCFDialect, ondrix::ondsp::OndspDialect>();
    target.addIllegalOp<ondrix::ondsp::ReduceMacOp>();

    if (failed(applyPartialConversion(getOperation(), target, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> ondrix::createConvertOndspToScalarPass() {
  return std::make_unique<ConvertOndspToScalarPass>();
}
