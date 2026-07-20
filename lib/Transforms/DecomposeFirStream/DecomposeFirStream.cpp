#include "ondrix/Transforms/Passes.h"

#include "ondrix/Dialect/ondrix/IR/OndrixDialect.h"
#include "ondrix/Dialect/ondrix/IR/OndrixOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace ondrix {
#define GEN_PASS_DEF_DECOMPOSEONDRIXFIRSTREAM
#include "ondrix/Transforms/Passes.h.inc"
} // namespace ondrix

using namespace mlir;

namespace {

static Value createEmptyTensor(Location loc, RankedTensorType type, Value dynamicLength,
                               OpBuilder &builder) {
  SmallVector<Value> dynamicSizes;
  if (type.isDynamicDim(0))
    dynamicSizes.push_back(dynamicLength);
  return builder.create<tensor::EmptyOp>(loc, type.getShape(), type.getElementType(), dynamicSizes);
}

static OpFoldResult getTensorLength(Value tensor, Value dynamicLength, OpBuilder &builder) {
  auto type = cast<RankedTensorType>(tensor.getType());
  if (!type.isDynamicDim(0))
    return builder.getIndexAttr(type.getDimSize(0));
  return dynamicLength;
}

static void assertStreamShape(ondrix::ir::FirStreamOp op, Value inputLength,
                              Value coefficientLength, Value stateLength, Value zero, Value one,
                              OpBuilder &builder) {
  Location loc = op.getLoc();
  Value hasCoefficients =
      builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ugt, coefficientLength, zero);
  builder.create<cf::AssertOp>(
      loc, hasCoefficients, builder.getStringAttr("FIR stream requires at least one coefficient"));
  Value expectedCoefficientLength = builder.create<arith::AddIOp>(loc, stateLength, one);
  Value stateMatches = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq,
                                                     expectedCoefficientLength, coefficientLength);
  builder.create<cf::AssertOp>(
      loc, stateMatches,
      builder.getStringAttr("FIR stream state length must equal coefficient length minus one"));

  Value extendedLength = builder.create<arith::AddIOp>(loc, stateLength, inputLength);
  Value extendedLengthFits =
      builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::uge, extendedLength, inputLength);
  builder.create<cf::AssertOp>(
      loc, extendedLengthFits,
      builder.getStringAttr("FIR stream history and input exceed the indexable extent range"));

  RankedTensorType inputType = op.getInput().getType();
  RankedTensorType outputType = op.getOutput().getType();
  if (inputType.isDynamicDim(0) && !outputType.isDynamicDim(0)) {
    Value expectedOutputLength =
        builder.create<arith::ConstantIndexOp>(loc, outputType.getDimSize(0));
    Value outputMatches = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, inputLength,
                                                        expectedOutputLength);
    builder.create<cf::AssertOp>(
        loc, outputMatches,
        builder.getStringAttr("FIR stream output length must equal input chunk length"));
  }

  RankedTensorType stateType = op.getState().getType();
  RankedTensorType nextStateType = op.getNextState().getType();
  if (stateType.isDynamicDim(0) && !nextStateType.isDynamicDim(0)) {
    Value expectedNextStateLength =
        builder.create<arith::ConstantIndexOp>(loc, nextStateType.getDimSize(0));
    Value nextStateMatches = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq,
                                                           stateLength, expectedNextStateLength);
    builder.create<cf::AssertOp>(
        loc, nextStateMatches,
        builder.getStringAttr("FIR stream next-state length must equal state length"));
  }
}

class DecomposeFirStreamPattern final : public OpRewritePattern<ondrix::ir::FirStreamOp> {
public:
  using OpRewritePattern<ondrix::ir::FirStreamOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ondrix::ir::FirStreamOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Value inputLength = rewriter.create<tensor::DimOp>(loc, op.getInput(), zero);
    Value coefficientLength = rewriter.create<tensor::DimOp>(loc, op.getCoeffs(), zero);
    Value stateLength = rewriter.create<tensor::DimOp>(loc, op.getState(), zero);
    assertStreamShape(op, inputLength, coefficientLength, stateLength, zero, one, rewriter);

    Value extendedLength = rewriter.create<arith::AddIOp>(loc, stateLength, inputLength);
    auto extendedType =
        RankedTensorType::get({ShapedType::kDynamic}, op.getInput().getType().getElementType());
    Value emptyExtended = createEmptyTensor(loc, extendedType, extendedLength, rewriter);
    SmallVector<OpFoldResult> zeroOffset{rewriter.getIndexAttr(0)};
    SmallVector<OpFoldResult> oneStride{rewriter.getIndexAttr(1)};
    OpFoldResult stateSize = getTensorLength(op.getState(), stateLength, rewriter);
    OpFoldResult inputSize = getTensorLength(op.getInput(), inputLength, rewriter);
    Value withState =
        rewriter.create<tensor::InsertSliceOp>(loc, op.getState(), emptyExtended, zeroOffset,
                                               ArrayRef<OpFoldResult>{stateSize}, oneStride);
    Value extended = rewriter.create<tensor::InsertSliceOp>(
        loc, op.getInput(), withState, ArrayRef<OpFoldResult>{stateLength},
        ArrayRef<OpFoldResult>{inputSize}, oneStride);

    RankedTensorType outputType = op.getOutput().getType();
    Value outputInit = createEmptyTensor(loc, outputType, inputLength, rewriter);
    Value hasInput =
        rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ugt, inputLength, zero);
    auto output = rewriter.create<scf::IfOp>(loc, TypeRange{outputType}, hasInput,
                                             /*withElseRegion=*/true);
    OpBuilder thenBuilder = output.getThenBodyBuilder();
    Value filtered = thenBuilder.create<ondrix::ir::FirFilterOp>(
        loc, outputType, extended, op.getCoeffs(), outputInit, Value(),
        ondrix::ir::FirBoundaryMode::Valid, op.getNumericAttr(), op.getProductAttr(),
        op.getAccumulatorAttr(), op.getDstAttr(), op.getRoundingAttr(), op.getOverflowAttr());
    thenBuilder.create<scf::YieldOp>(loc, filtered);
    OpBuilder elseBuilder = output.getElseBodyBuilder();
    elseBuilder.create<scf::YieldOp>(loc, outputInit);

    RankedTensorType nextStateType = op.getNextState().getType();
    OpFoldResult nextStateSize =
        nextStateType.isDynamicDim(0)
            ? OpFoldResult(stateLength)
            : OpFoldResult(rewriter.getIndexAttr(nextStateType.getDimSize(0)));
    Value nextState = rewriter.create<tensor::ExtractSliceOp>(
        loc, nextStateType, extended, ArrayRef<OpFoldResult>{inputLength},
        ArrayRef<OpFoldResult>{nextStateSize}, oneStride);
    rewriter.replaceOp(op, ValueRange{output.getResult(0), nextState});
    return success();
  }
};

class DecomposeOndrixFirStreamPass final
    : public ondrix::impl::DecomposeOndrixFirStreamBase<DecomposeOndrixFirStreamPass> {
public:
  using ondrix::impl::DecomposeOndrixFirStreamBase<
      DecomposeOndrixFirStreamPass>::DecomposeOndrixFirStreamBase;

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<DecomposeFirStreamPattern>(&getContext());
    if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> ondrix::createDecomposeOndrixFirStreamPass() {
  return std::make_unique<DecomposeOndrixFirStreamPass>();
}
