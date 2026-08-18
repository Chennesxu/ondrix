#include "ondrix/Dialect/ortumcore/IR/OrtumCoreOps.h"

#include "mlir/IR/Diagnostics.h"

using namespace mlir;
using namespace ondrix::ortumcore;

LogicalResult AccOutOp::verify() {
  int64_t shift = getShift();
  if (shift < 0 || shift > 15)
    return emitOpError("accumulator readout shift must lie in [0, 15]");
  return success();
}

template <typename OpTy> static LogicalResult verifyScaledBinaryShift(OpTy op) {
  int64_t shift = op.getShift();
  if (shift < 0 || shift > 3)
    return op.emitOpError("scaled saturating add/sub shift must lie in [0, 3]");
  return success();
}

LogicalResult SatShiftAddOp::verify() { return verifyScaledBinaryShift(*this); }

LogicalResult SatShiftSubOp::verify() { return verifyScaledBinaryShift(*this); }

LogicalResult CxMulConjOp::verify() {
  int64_t shift = getShift();
  if (shift < 0 || shift > 31)
    return emitOpError("packed complex product shift must lie in [0, 31]");
  return success();
}

LogicalResult CxBflyOp::verify() {
  int64_t shift = getShift();
  if (shift < 0 || shift > 1)
    return emitOpError("packed complex butterfly shift must lie in [0, 1]");
  return success();
}

#define GET_OP_CLASSES
#include "ondrix/Dialect/ortumcore/IR/OrtumCoreOps.cpp.inc"
