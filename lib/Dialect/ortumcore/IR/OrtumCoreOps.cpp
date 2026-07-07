#include "ondrix/Dialect/ortumcore/IR/OrtumCoreOps.h"

#include "mlir/IR/Diagnostics.h"

using namespace mlir;
using namespace ondrix::ortumcore;

LogicalResult FftTrivialStageOp::verify() {
  auto variant = getOperation()->getAttrOfType<IntegerAttr>("variant");
  if (!variant)
    return emitOpError("requires i64 variant attribute");

  int64_t variantValue = variant.getInt();
  if (variantValue < 1 || variantValue > 10)
    return emitOpError("requires variant in range [1, 10]");

  return success();
}

#define GET_OP_CLASSES
#include "ondrix/Dialect/ortumcore/IR/OrtumCoreOps.cpp.inc"
