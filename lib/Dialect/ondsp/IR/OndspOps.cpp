#include "ondrix/Dialect/ondsp/IR/OndspOps.h"

#include "mlir/Interfaces/SideEffectInterfaces.h"

#include <limits>
#include <optional>

using namespace mlir;
using namespace ondrix::ondsp;

#define GET_OP_CLASSES
#include "ondrix/Dialect/ondsp/IR/OndspOps.cpp.inc"

namespace {

static void addMemRefReadEffect(Value value,
                                SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  if (!isa<BaseMemRefType>(value.getType()))
    return;
  effects.emplace_back(MemoryEffects::Read::get(), value, SideEffects::DefaultResource::get());
}

static LogicalResult verifyValueOnlyTypes(Operation *op) {
  auto containsMemRef = [](Type type) {
    return type.walk([](BaseMemRefType) { return WalkResult::interrupt(); }).wasInterrupted();
  };
  if (llvm::any_of(op->getOperandTypes(), containsMemRef))
    return op->emitOpError("value-only operation does not accept memref operands");
  if (llvm::any_of(op->getResultTypes(), containsMemRef))
    return op->emitOpError("value-only operation does not produce memref results");
  return success();
}

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
  if (accumulator.getSignedness() != numeric.getSignedness())
    return op->emitOpError("accumulator signedness must match the fixed numeric policy");

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

static LogicalResult verifyButterflyPolicies(Operation *op, Attribute numeric,
                                             std::optional<ProductAttr> product,
                                             std::optional<ScaleAttr> scale) {
  if (failed(verifyOptionalProductPolicy(op, numeric, product)))
    return failure();

  if (isa<FixedAttr>(numeric)) {
    if (!scale)
      return op->emitOpError("fixed numeric butterfly requires a scale attribute");
    return success();
  }

  if (scale)
    return op->emitOpError("floating-point numeric butterfly must not specify a scale attribute");
  return success();
}

static bool isPackedI16Layout(CxLayoutAttr layout) {
  return layout.getLayout() == ComplexLayout::PackedI16ImagHiRealLo ||
         layout.getLayout() == ComplexLayout::PackedI16RealHiImagLo;
}

static bool hasI32Container(Type type) {
  if (auto integer = type.dyn_cast<IntegerType>())
    return integer.getWidth() == 32;
  if (auto shaped = type.dyn_cast<ShapedType>()) {
    auto element = shaped.getElementType().dyn_cast<IntegerType>();
    return element && element.getWidth() == 32;
  }
  return false;
}

} // namespace

void ReduceMacOp::getEffects(SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  addMemRefReadEffect(getLhs(), effects);
  addMemRefReadEffect(getRhs(), effects);
}

Speculation::Speculatability ReduceMacOp::getSpeculatability() {
  return (isa<BaseMemRefType>(getLhs().getType()) || isa<BaseMemRefType>(getRhs().getType()))
             ? Speculation::NotSpeculatable
             : Speculation::Speculatable;
}

LogicalResult AssumeNumericOp::verify() {
  if (failed(verifyValueOnlyTypes(*this)))
    return failure();
  return verifyValueNumericType(*this, getInput().getType(), getNumeric(), "input");
}

LogicalResult ConvertOp::verify() {
  if (failed(verifyValueOnlyTypes(*this)))
    return failure();
  if (failed(verifyValueNumericType(*this, getInput().getType(), getSrc(), "input")))
    return failure();
  return verifyValueNumericType(*this, getResult().getType(), getDst(), "result");
}

LogicalResult RoundShiftOp::verify() { return verifyValueOnlyTypes(*this); }

LogicalResult SatCastOp::verify() { return verifyValueOnlyTypes(*this); }

LogicalResult SatAddShiftOp::verify() { return verifyValueOnlyTypes(*this); }

LogicalResult SatSubShiftOp::verify() { return verifyValueOnlyTypes(*this); }

LogicalResult AccInitOp::verify() { return verifyValueOnlyTypes(*this); }

LogicalResult MacOp::verify() {
  return verifyMacLike(*this, getAcc(), getLhs(), getRhs(), getNumeric(), getProduct());
}

LogicalResult MacSubOp::verify() {
  return verifyMacLike(*this, getAcc(), getLhs(), getRhs(), getNumeric(), getProduct());
}

LogicalResult AccExtractOp::verify() {
  if (auto scale = getScale()) {
    if (scale->getSaturateTo() != getResult().getType())
      return emitOpError("scale saturate_to type must match the result type");
  }
  return success();
}

LogicalResult AccExportOp::verify() {
  AccType accumulator = getAcc().getType();
  FixedAttr destination = getDst();
  if (accumulator.getSignedness() != destination.getSignedness())
    return emitOpError("accumulator and destination signedness must match");
  if (accumulator.getFrac() < destination.getFrac())
    return emitOpError("destination frac must not exceed accumulator frac");
  if (getResult().getType() != destination.getStorage())
    return emitOpError("result type must match destination storage type");
  return success();
}

LogicalResult ReduceMacOp::verify() {
  if (failed(verifyOptionalProductPolicy(*this, getNumeric(), getProduct())))
    return failure();

  if (auto fixed = dyn_cast<FixedAttr>(getNumeric())) {
    if (failed(verifyFixedStorageOperands(*this, fixed, getLhs(), getRhs())))
      return failure();
    auto resultType = getResult().getType().dyn_cast<IntegerType>();
    if (!resultType || resultType.getWidth() < 32)
      return emitOpError("fixed reduce_mac result must be an integer type of at least 32 bits");
    return success();
  }
  if (failed(verifyValueNumericType(*this, getLhs().getType(), getNumeric(), "lhs")) ||
      failed(verifyValueNumericType(*this, getRhs().getType(), getNumeric(), "rhs")))
    return failure();
  return verifyValueNumericType(*this, getResult().getType(), getNumeric(), "result");
}

LogicalResult CxMulOp::verify() { return verifyValueOnlyTypes(*this); }

LogicalResult CxButterflyOp::verify() {
  if (failed(verifyValueOnlyTypes(*this)))
    return failure();
  if (failed(verifyButterflyPolicies(*this, getNumeric(), getProduct(), getScale())))
    return failure();

  if (!isPackedI16Layout(getLayout())) {
    for (auto [index, type] : llvm::enumerate(getOperandTypes())) {
      if (failed(verifyValueNumericType(*this, type, getNumeric(),
                                        index == 0   ? "a"
                                        : index == 1 ? "b"
                                                     : "twiddle")))
        return failure();
    }
    for (auto [index, type] : llvm::enumerate(getResultTypes())) {
      if (failed(verifyValueNumericType(*this, type, getNumeric(), index == 0 ? "out0" : "out1")))
        return failure();
    }
    return success();
  }

  auto fixed = dyn_cast<FixedAttr>(getNumeric());
  auto storage = fixed ? fixed.getStorage().dyn_cast<IntegerType>() : IntegerType();
  if (!storage || storage.getWidth() != 16)
    return emitOpError("packed i16 complex layout requires an i16 fixed numeric policy");

  for (Type type : getOperandTypes()) {
    if (!hasI32Container(type))
      return emitOpError("packed i16 butterfly operands must use i32 container storage");
  }
  for (Type type : getResultTypes()) {
    if (!hasI32Container(type))
      return emitOpError("packed i16 butterfly results must use i32 container storage");
  }
  return success();
}

LogicalResult FftStageOp::verify() { return verifyValueOnlyTypes(*this); }
