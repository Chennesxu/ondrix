#include "ondrix/Dialect/ondsp/IR/OndspOps.h"

#include <limits>
#include <optional>

using namespace mlir;
using namespace ondrix::ondsp;

#define GET_OP_CLASSES
#include "ondrix/Dialect/ondsp/IR/OndspOps.cpp.inc"

namespace {

static bool hasStorageType(Type type, Type storage) {
  if (type == storage)
    return true;

  if (auto shaped = type.dyn_cast<ShapedType>())
    return shaped.getElementType() == storage;

  return false;
}

static LogicalResult verifyValueNumericType(Operation *op, Type type, Attribute numeric,
                                            StringRef valueName) {
  Type storage;
  if (auto fixed = dyn_cast<FixedAttr>(numeric))
    storage = fixed.getStorage();
  else if (auto fp = dyn_cast<FpAttr>(numeric))
    storage = fp.getFormat();
  else
    return op->emitOpError("requires a fixed-point or floating-point numeric attribute");

  if (!hasStorageType(type, storage))
    return op->emitOpError() << valueName << " type does not match numeric storage type";
  return success();
}

static LogicalResult verifyFixedStorageOperands(Operation *op, FixedAttr numeric, Value lhs,
                                                Value rhs) {
  if (!hasStorageType(lhs.getType(), numeric.getStorage()))
    return op->emitOpError("lhs type does not match fixed numeric storage type");
  if (!hasStorageType(rhs.getType(), numeric.getStorage()))
    return op->emitOpError("rhs type does not match fixed numeric storage type");
  return success();
}

static FailureOr<unsigned> getProductFrac(Operation *op, FixedAttr numeric, ProductAttr product) {
  auto storage = numeric.getStorage().cast<IntegerType>();
  uint64_t frac = numeric.getFrac();
  uint64_t width = storage.getWidth();
  uint64_t doubledFrac = frac * 2;

  if (frac > std::numeric_limits<unsigned>::max() / 2)
    return op->emitOpError("product fractional position is unrepresentable");

  if (product.getSelection() == ProductSelection::High) {
    if (doubledFrac < width)
      return op->emitOpError("high product fractional position would be negative");
    doubledFrac -= width;
  }

  if (doubledFrac > std::numeric_limits<unsigned>::max())
    return op->emitOpError("product fractional position is unrepresentable");
  return static_cast<unsigned>(doubledFrac);
}

static StringRef getProductName(ProductAttr product) {
  return product.getSelection() == ProductSelection::Full ? "full" : "high";
}

static LogicalResult verifyMacLike(Operation *op, Value acc, Value lhs, Value rhs,
                                   FixedAttr numeric, ProductAttr product) {
  if (failed(verifyFixedStorageOperands(op, numeric, lhs, rhs)))
    return failure();

  auto accumulator = acc.getType().cast<AccType>();
  FailureOr<unsigned> expectedFrac = getProductFrac(op, numeric, product);
  if (failed(expectedFrac))
    return failure();

  if (accumulator.getFrac() != *expectedFrac)
    return op->emitOpError() << "accumulator frac " << accumulator.getFrac()
                             << " does not match expected frac " << *expectedFrac << " for "
                             << getProductName(product) << " product";
  return success();
}

static LogicalResult verifyOptionalProductPolicy(Operation *op, Attribute numeric,
                                                 std::optional<ProductAttr> product) {
  if (isa<FixedAttr>(numeric)) {
    if (!product)
      return op->emitOpError("fixed numeric policy requires a product attribute");
    return success();
  }

  if (product)
    return op->emitOpError("floating-point numeric policy must not specify a product attribute");
  return success();
}

} // namespace

LogicalResult AssumeNumericOp::verify() {
  return verifyValueNumericType(*this, getInput().getType(), getNumeric(), "input");
}

LogicalResult ConvertOp::verify() {
  if (failed(verifyValueNumericType(*this, getInput().getType(), getSrc(), "input")))
    return failure();
  return verifyValueNumericType(*this, getResult().getType(), getDst(), "result");
}

LogicalResult MacOp::verify() {
  return verifyMacLike(*this, getAcc(), getLhs(), getRhs(), getNumeric(), getProduct());
}

LogicalResult MacSubOp::verify() {
  return verifyMacLike(*this, getAcc(), getLhs(), getRhs(), getNumeric(), getProduct());
}

LogicalResult ReduceMacOp::verify() {
  if (failed(verifyOptionalProductPolicy(*this, getNumeric(), getProduct())))
    return failure();

  if (auto fixed = dyn_cast<FixedAttr>(getNumeric()))
    return verifyFixedStorageOperands(*this, fixed, getLhs(), getRhs());
  if (failed(verifyValueNumericType(*this, getLhs().getType(), getNumeric(), "lhs")) ||
      failed(verifyValueNumericType(*this, getRhs().getType(), getNumeric(), "rhs")))
    return failure();
  return verifyValueNumericType(*this, getResult().getType(), getNumeric(), "result");
}

LogicalResult CxButterflyOp::verify() {
  return verifyOptionalProductPolicy(*this, getNumeric(), getProduct());
}
