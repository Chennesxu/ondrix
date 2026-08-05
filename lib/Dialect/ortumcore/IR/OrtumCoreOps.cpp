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

#define GET_OP_CLASSES
#include "ondrix/Dialect/ortumcore/IR/OrtumCoreOps.cpp.inc"
