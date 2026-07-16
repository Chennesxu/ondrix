#include "ondrix/Dialect/ondsp/IR/OndspOps.h"
#include "ondrix/Dialect/ondsp/IR/OndspSemantics.h"
#include "ondrix/Support/DSPTypeUtils.h"

#include "mlir/Interfaces/SideEffectInterfaces.h"

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
  if (llvm::any_of(op->getOperandTypes(), ondrix::containsScalableVectorType))
    return op->emitOpError("value-only operation does not accept scalable vector operands");
  if (llvm::any_of(op->getResultTypes(), ondrix::containsScalableVectorType))
    return op->emitOpError("value-only operation does not produce scalable vector results");
  if (llvm::any_of(op->getOperandTypes(), ondrix::containsDynamicOrUnrankedShapedType))
    return op->emitOpError(
        "value-only operation does not accept dynamic or unranked shaped operands");
  if (llvm::any_of(op->getResultTypes(), ondrix::containsDynamicOrUnrankedShapedType))
    return op->emitOpError(
        "value-only operation does not produce dynamic or unranked shaped results");
  return success();
}

static bool hasStorageType(Type type, Type storage) {
  if (type == storage)
    return true;

  if (auto shaped = type.dyn_cast<ShapedType>())
    return shaped.getElementType() == storage;

  return false;
}

static Type getNumericStorage(Attribute numeric) {
  if (auto fixed = dyn_cast<FixedAttr>(numeric))
    return fixed.getStorage();
  return cast<FpAttr>(numeric).getFormat();
}

static LogicalResult verifyValueNumericType(Operation *op, Type type, Attribute numeric,
                                            StringRef valueName) {
  if (!isa<FixedAttr, FpAttr>(numeric))
    return op->emitOpError("requires a fixed-point or floating-point numeric attribute");

  Type storage = getNumericStorage(numeric);
  if (!hasStorageType(type, storage))
    return op->emitOpError() << valueName << " type does not match numeric storage type";
  return success();
}

static LogicalResult verifySameElementwiseShape(Operation *op, TypeRange types) {
  if (types.empty())
    return success();
  Type reference = types.front();
  if (llvm::all_of(types.drop_front(),
                   [&](Type type) { return ondrix::haveSameElementwiseShape(reference, type); }))
    return success();
  return op->emitOpError("operands and results must use the same scalar or static shaped domain");
}

static LogicalResult verifyResultElementType(Operation *op, Type result, Type expected) {
  if (ondrix::getElementTypeOrSelf(result) == expected)
    return success();
  return op->emitOpError("result element type does not match the destination storage type");
}

static LogicalResult verifySignlessIntegerElementType(Operation *op, Type type,
                                                      StringRef valueName) {
  auto elementType = dyn_cast<IntegerType>(ondrix::getElementTypeOrSelf(type));
  if (!elementType || !elementType.isSignless())
    return op->emitOpError() << valueName << " must use a signless integer element type";
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

  FailureOr<unsigned> expectedFrac = inferProductFractionalBits(op, numeric, product);
  if (failed(expectedFrac))
    return failure();

  if (accumulator.getFrac() != *expectedFrac)
    return op->emitOpError() << "accumulator frac " << accumulator.getFrac()
                             << " does not match expected frac " << *expectedFrac << " for "
                             << getProductName(product) << " product";
  return success();
}

static LogicalResult verifyButterflyPolicies(Operation *op, Attribute numeric,
                                             std::optional<ProductAttr> product,
                                             std::optional<ScaleAttr> scale) {
  if (failed(verifyProductPolicy(op, numeric, product)))
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
    return integer.isSignless() && integer.getWidth() == 32;
  if (auto shaped = type.dyn_cast<ShapedType>()) {
    auto element = shaped.getElementType().dyn_cast<IntegerType>();
    return element && element.isSignless() && element.getWidth() == 32;
  }
  return false;
}

static LogicalResult verifyReduceDomain(ReduceMacOp op) {
  auto lhs = dyn_cast<ShapedType>(op.getLhs().getType());
  auto rhs = dyn_cast<ShapedType>(op.getRhs().getType());
  if (!lhs || !rhs || !lhs.hasRank() || !rhs.hasRank() || lhs.getRank() != 1 || rhs.getRank() != 1)
    return op.emitOpError("shaped operands must be rank-1");
  if (ondrix::isScalableVectorType(op.getLhs().getType()) ||
      ondrix::isScalableVectorType(op.getRhs().getType()))
    return op.emitOpError("scalable vector operands are not supported");
  if (lhs.getElementType() != rhs.getElementType())
    return op.emitOpError("shaped operand element types must match");

  int64_t lhsLength = lhs.getDimSize(0);
  int64_t rhsLength = rhs.getDimSize(0);
  if (!ShapedType::isDynamic(lhsLength) && !ShapedType::isDynamic(rhsLength) &&
      lhsLength != rhsLength)
    return op.emitOpError("shaped operands must have equal static lengths");
  return success();
}

} // namespace

void ReduceMacOp::getEffects(SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  addMemRefReadEffect(getLhs(), effects);
  addMemRefReadEffect(getRhs(), effects);
}

Speculation::Speculatability ReduceMacOp::getSpeculatability() {
  return (ondrix::requiresConservativeDSPSpeculation(getLhs().getType()) ||
          ondrix::requiresConservativeDSPSpeculation(getRhs().getType()))
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
  if (failed(verifySameElementwiseShape(*this, {getInput().getType(), getResult().getType()})))
    return failure();
  if (failed(verifyValueNumericType(*this, getInput().getType(), getSrc(), "input")))
    return failure();
  return verifyValueNumericType(*this, getResult().getType(), getDst(), "result");
}

LogicalResult RoundShiftOp::verify() {
  if (failed(verifyValueOnlyTypes(*this)))
    return failure();
  if (failed(verifySameElementwiseShape(*this, {getInput().getType(), getResult().getType()})))
    return failure();
  if (failed(verifySignlessIntegerElementType(*this, getInput().getType(), "input")))
    return failure();
  return verifyResultElementType(*this, getResult().getType(), getScale().getSaturateTo());
}

LogicalResult SatCastOp::verify() {
  if (failed(verifyValueOnlyTypes(*this)))
    return failure();
  if (failed(verifySameElementwiseShape(*this, {getInput().getType(), getResult().getType()})))
    return failure();
  Type destination = getNumericStorage(getNumeric());
  return verifyResultElementType(*this, getResult().getType(), destination);
}

static LogicalResult verifyBinaryShiftValueDomain(Operation *op, Value lhs, Value rhs, Value result,
                                                  ScaleAttr scale) {
  if (failed(verifySameElementwiseShape(op, {lhs.getType(), rhs.getType(), result.getType()})))
    return failure();
  if (failed(verifySignlessIntegerElementType(op, lhs.getType(), "lhs")) ||
      failed(verifySignlessIntegerElementType(op, rhs.getType(), "rhs")))
    return failure();
  if (ondrix::getElementTypeOrSelf(lhs.getType()) != ondrix::getElementTypeOrSelf(rhs.getType()))
    return op->emitOpError("lhs and rhs element types must match");
  return verifyResultElementType(op, result.getType(), scale.getSaturateTo());
}

LogicalResult SatAddShiftOp::verify() {
  if (failed(verifyValueOnlyTypes(*this)))
    return failure();
  return verifyBinaryShiftValueDomain(*this, getLhs(), getRhs(), getResult(), getScale());
}

LogicalResult SatSubShiftOp::verify() {
  if (failed(verifyValueOnlyTypes(*this)))
    return failure();
  return verifyBinaryShiftValueDomain(*this, getLhs(), getRhs(), getResult(), getScale());
}

LogicalResult AccImportOp::verify() {
  FixedAttr source = getSrc();
  AccType accumulator = getAcc().getType();
  if (getInput().getType() != source.getStorage())
    return emitOpError("input type must match source storage type");
  if (accumulator.getSignedness() != source.getSignedness())
    return emitOpError("source and accumulator signedness must match");
  if (accumulator.getFrac() < source.getFrac())
    return emitOpError("exact import requires accumulator frac to be at least source frac");

  unsigned sourceWidth = source.getStorage().cast<IntegerType>().getWidth();
  unsigned accumulatorWidth = accumulator.getStorage().cast<IntegerType>().getWidth();
  uint64_t requiredWidth =
      static_cast<uint64_t>(sourceWidth) + accumulator.getFrac() - source.getFrac();
  if (requiredWidth > accumulatorWidth)
    return emitOpError() << "exact import requires at least " << requiredWidth
                         << " accumulator storage bits";
  return success();
}

LogicalResult MacOp::verify() {
  return verifyMacLike(*this, getAcc(), getLhs(), getRhs(), getNumeric(), getProduct());
}

LogicalResult MacSubOp::verify() {
  return verifyMacLike(*this, getAcc(), getLhs(), getRhs(), getNumeric(), getProduct());
}

LogicalResult AccAddTermOp::verify() {
  AccType accumulator = getAcc().getType();
  FixedAttr termNumeric = getTermNumeric();
  if (getTerm().getType() != termNumeric.getStorage())
    return emitOpError("term type must match term numeric storage type");
  if (accumulator.getSignedness() != termNumeric.getSignedness())
    return emitOpError("term and accumulator signedness must match");
  if (accumulator.getFrac() != termNumeric.getFrac())
    return emitOpError("term and accumulator frac must match");
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
  if (failed(verifyReduceDomain(*this)))
    return failure();
  if (failed(verifyProductPolicy(*this, getNumeric(), getProduct())))
    return failure();

  if (auto fixed = dyn_cast<FixedAttr>(getNumeric())) {
    auto accumulator = dyn_cast<AccType>(getInitial().getType());
    if (!accumulator)
      return emitOpError("fixed reduce_mac initial and result must use !ondsp.acc");
    return verifyMacLike(*this, getInitial(), getLhs(), getRhs(), fixed, *getProduct());
  }
  if (failed(verifyValueNumericType(*this, getLhs().getType(), getNumeric(), "lhs")) ||
      failed(verifyValueNumericType(*this, getRhs().getType(), getNumeric(), "rhs")))
    return failure();
  auto fp = cast<FpAttr>(getNumeric());
  if (getInitial().getType() != fp.getFormat())
    return emitOpError("floating-point reduce_mac initial and result must match numeric format");
  return success();
}

static LogicalResult verifyComplexValueDomain(Operation *op, TypeRange types, CxLayoutAttr layout,
                                              Attribute numeric) {
  if (failed(verifySameElementwiseShape(op, types)))
    return failure();
  if (!isPackedI16Layout(layout)) {
    for (Type type : types)
      if (failed(verifyValueNumericType(op, type, numeric, "complex value")))
        return failure();
    return success();
  }

  auto fixed = dyn_cast<FixedAttr>(numeric);
  auto storage = fixed ? dyn_cast<IntegerType>(fixed.getStorage()) : IntegerType();
  if (!storage || storage.getWidth() != 16)
    return op->emitOpError("packed i16 complex layout requires an i16 fixed numeric policy");
  if (!llvm::all_of(types, hasI32Container))
    return op->emitOpError("packed i16 complex values require signless i32 container storage");
  return success();
}

LogicalResult CxMulOp::verify() {
  if (failed(verifyValueOnlyTypes(*this)))
    return failure();
  return verifyComplexValueDomain(*this,
                                  {getLhs().getType(), getRhs().getType(), getResult().getType()},
                                  getLayout(), getNumeric());
}

LogicalResult CxButterflyOp::verify() {
  if (failed(verifyValueOnlyTypes(*this)))
    return failure();
  if (failed(verifyButterflyPolicies(*this, getNumeric(), getProduct(), getScale())))
    return failure();

  SmallVector<Type> valueTypes(getOperandTypes());
  valueTypes.append(getResultTypes().begin(), getResultTypes().end());
  if (failed(verifySameElementwiseShape(*this, valueTypes)))
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
      return emitOpError("packed i16 butterfly operands must use signless i32 container storage");
  }
  for (Type type : getResultTypes()) {
    if (!hasI32Container(type))
      return emitOpError("packed i16 butterfly results must use signless i32 container storage");
  }
  if (getScale()->getSaturateTo() != fixed.getStorage())
    return emitOpError("packed i16 butterfly saturate_to must match fixed numeric storage type");
  return success();
}

LogicalResult FftStageOp::verify() { return verifyValueOnlyTypes(*this); }
