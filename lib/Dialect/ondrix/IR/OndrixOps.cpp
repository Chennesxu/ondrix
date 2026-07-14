#include "ondrix/Dialect/ondrix/IR/OndrixOps.h"

#include "ondrix/Dialect/ondsp/IR/OndspAttrs.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include <optional>

using namespace mlir;
using namespace ondrix::ir;

#define GET_OP_CLASSES
#include "ondrix/Dialect/ondrix/IR/OndrixOps.cpp.inc"

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

static LogicalResult verifyButterflyPolicies(Operation *op, Attribute numeric,
                                             std::optional<ondrix::ondsp::ProductAttr> product,
                                             std::optional<ondrix::ondsp::ScaleAttr> scale) {
  if (failed(verifyOptionalProductPolicy(op, numeric, product)))
    return failure();

  if (isa<ondrix::ondsp::FixedAttr>(numeric)) {
    if (!scale)
      return op->emitOpError("fixed numeric butterfly requires a scale attribute");
    return success();
  }

  if (scale)
    return op->emitOpError("floating-point numeric butterfly must not specify a scale attribute");
  return success();
}

} // namespace

void FirOp::getEffects(SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  addMemRefReadEffect(getInput(), effects);
  addMemRefReadEffect(getCoeffs(), effects);
}

Speculation::Speculatability FirOp::getSpeculatability() {
  return (isa<BaseMemRefType>(getInput().getType()) || isa<BaseMemRefType>(getCoeffs().getType()))
             ? Speculation::NotSpeculatable
             : Speculation::Speculatable;
}

void DotOp::getEffects(SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  addMemRefReadEffect(getLhs(), effects);
  addMemRefReadEffect(getRhs(), effects);
}

Speculation::Speculatability DotOp::getSpeculatability() {
  return (isa<BaseMemRefType>(getLhs().getType()) || isa<BaseMemRefType>(getRhs().getType()))
             ? Speculation::NotSpeculatable
             : Speculation::Speculatable;
}

LogicalResult FirOp::verify() {
  return verifyOptionalProductPolicy(*this, getNumeric(), getProduct());
}

LogicalResult DotOp::verify() {
  return verifyOptionalProductPolicy(*this, getNumeric(), getProduct());
}

LogicalResult ButterflyOp::verify() {
  if (failed(verifyValueOnlyTypes(*this)))
    return failure();
  return verifyButterflyPolicies(*this, getNumeric(), getProduct(), getScale());
}

LogicalResult QuantizeOp::verify() { return verifyValueOnlyTypes(*this); }
