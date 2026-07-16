#include "ondrix/Conversion/Utils/ReductionUtils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

using namespace mlir;

namespace ondrix::conversion {
namespace {

FailureOr<MemRefType> getRankOneMemRefType(Operation *op, Value value, Type elementType,
                                           StringRef operandName, StringRef consumer) {
  auto type = dyn_cast<MemRefType>(value.getType());
  if (!type || type.getRank() != 1 || type.getElementType() != elementType)
    return op->emitOpError() << operandName << " must be a rank-1 memref with " << elementType
                             << " elements for " << consumer;
  return type;
}

Value getDimZeroSize(Location loc, Value value, MemRefType type, Value zeroIndex,
                     OpBuilder &builder) {
  if (!type.isDynamicDim(0))
    return builder.create<arith::ConstantIndexOp>(loc, type.getDimSize(0));
  return builder.create<memref::DimOp>(loc, value, zeroIndex);
}

} // namespace

FailureOr<RankOneReductionBounds> createRankOneMemRefReductionBounds(Operation *op, Value lhs,
                                                                     Value rhs, Type elementType,
                                                                     StringRef consumer,
                                                                     OpBuilder &builder) {
  FailureOr<MemRefType> lhsType = getRankOneMemRefType(op, lhs, elementType, "lhs", consumer);
  if (failed(lhsType))
    return failure();
  FailureOr<MemRefType> rhsType = getRankOneMemRefType(op, rhs, elementType, "rhs", consumer);
  if (failed(rhsType))
    return failure();

  if (!lhsType->isDynamicDim(0) && !rhsType->isDynamicDim(0) &&
      lhsType->getDimSize(0) != rhsType->getDimSize(0))
    return op->emitOpError("requires lhs and rhs to have equal static lengths");

  Location loc = op->getLoc();
  Value lowerBound = builder.create<arith::ConstantIndexOp>(loc, 0);
  Value upperBound = getDimZeroSize(loc, lhs, *lhsType, lowerBound, builder);
  if (lhsType->isDynamicDim(0) || rhsType->isDynamicDim(0)) {
    Value rhsSize = getDimZeroSize(loc, rhs, *rhsType, lowerBound, builder);
    Value lengthsMatch =
        builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, upperBound, rhsSize);
    builder.create<cf::AssertOp>(
        loc, lengthsMatch,
        builder.getStringAttr("ondsp.reduce_mac requires equal operand lengths"));
  }

  return RankOneReductionBounds{*lhsType, *rhsType, lowerBound, upperBound};
}

} // namespace ondrix::conversion
