#include "OndrixToOndspCommon.h"

#include "ondrix/Dialect/ondsp/IR/OndspOps.h"
#include "ondrix/Dialect/ondsp/IR/OndspSemantics.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

using namespace mlir;

namespace ondrix::conversion {

// Contract-invariant sites: a bare product has no addend to fuse, and these
// sums are built in declared order even where they form a reduction tree. A
// permission that is not used is a permission that is not emitted.
Value createFpMultiply(Location loc, Value lhs, Value rhs, OpBuilder &builder) {
  return builder.create<arith::MulFOp>(loc, lhs, rhs);
}

Value createFpAdd(Location loc, Value lhs, Value rhs, OpBuilder &builder) {
  return builder.create<arith::AddFOp>(loc, lhs, rhs);
}

Value createEmptyTensor(Location loc, RankedTensorType type, Value dynamicLength,
                        OpBuilder &builder) {
  SmallVector<Value> dynamicSizes;
  if (type.isDynamicDim(0))
    dynamicSizes.push_back(dynamicLength);
  return builder.create<tensor::EmptyOp>(loc, type.getShape(), type.getElementType(), dynamicSizes);
}

ondrix::ondsp::ScaleAttr getNearestEvenSaturatingShift(MLIRContext *context, unsigned shift) {
  return ondrix::ondsp::ScaleAttr::get(context, /*preShiftLeft=*/0, /*postShiftRight=*/shift,
                                       ondrix::ondsp::RoundingMode::NearestEven,
                                       ondrix::ondsp::OverflowMode::Saturate,
                                       IntegerType::get(context, 16));
}

} // namespace ondrix::conversion
