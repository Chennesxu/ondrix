#include "ondrix/Support/FirStreamRuntimeShape.h"

#include "ondrix/Dialect/ondrix/IR/OndrixOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"

using namespace mlir;

void ondrix::emitFirStreamRuntimeShapeAssertions(ir::FirStreamOp op, Value inputLength,
                                                 Value coefficientLength, Value stateLength,
                                                 Value zero, Value one, OpBuilder &builder) {
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
