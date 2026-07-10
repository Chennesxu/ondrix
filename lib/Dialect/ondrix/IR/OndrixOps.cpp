#include "ondrix/Dialect/ondrix/IR/OndrixOps.h"

#include "ondrix/Dialect/ondsp/IR/OndspAttrs.h"

#include <optional>

using namespace mlir;
using namespace ondrix::ir;

#define GET_OP_CLASSES
#include "ondrix/Dialect/ondrix/IR/OndrixOps.cpp.inc"

namespace {

static LogicalResult
verifyOptionalProductPolicy(Operation *op, Attribute numeric,
                            std::optional<ondrix::ondsp::ProductAttr> product) {
  if (isa<ondrix::ondsp::FixedAttr>(numeric)) {
    if (!product)
      return op->emitOpError("fixed numeric policy requires a product attribute");
    return success();
  }

  if (product)
    return op->emitOpError("floating-point numeric policy must not specify a product attribute");
  return success();
}

} // namespace

LogicalResult FirOp::verify() {
  return verifyOptionalProductPolicy(*this, getNumeric(), getProduct());
}

LogicalResult DotOp::verify() {
  return verifyOptionalProductPolicy(*this, getNumeric(), getProduct());
}

LogicalResult ButterflyOp::verify() {
  return verifyOptionalProductPolicy(*this, getNumeric(), getProduct());
}
