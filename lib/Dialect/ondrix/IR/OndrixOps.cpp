#include "ondrix/Dialect/ondrix/IR/OndrixOps.h"

#include "ondrix/Dialect/ondsp/IR/OndspAttrs.h"
#include "ondrix/Support/DSPTypeUtils.h"

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

static Type getNumericStorage(Attribute numeric) {
  if (auto fixed = dyn_cast<ondrix::ondsp::FixedAttr>(numeric))
    return fixed.getStorage();
  return cast<ondrix::ondsp::FpAttr>(numeric).getFormat();
}

static bool isPackedI16Layout(ondrix::ondsp::CxLayoutAttr layout) {
  return layout.getLayout() == ondrix::ondsp::ComplexLayout::PackedI16ImagHiRealLo ||
         layout.getLayout() == ondrix::ondsp::ComplexLayout::PackedI16RealHiImagLo;
}

static bool hasSignlessI32Element(Type type) {
  auto element = dyn_cast<IntegerType>(ondrix::getElementTypeOrSelf(type));
  return element && element.isSignless() && element.getWidth() == 32;
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

static LogicalResult verifyButterflyValueDomain(ButterflyOp op) {
  SmallVector<Type> types(op.getOperandTypes());
  llvm::append_range(types, op.getResultTypes());
  if (failed(verifySameElementwiseShape(op, types)))
    return failure();

  if (!isPackedI16Layout(op.getLayout())) {
    Type storage = getNumericStorage(op.getNumeric());
    if (!llvm::all_of(types,
                      [&](Type type) { return ondrix::getElementTypeOrSelf(type) == storage; }))
      return op.emitOpError("operand and result element types must match numeric storage type");
    return success();
  }

  auto fixed = dyn_cast<ondrix::ondsp::FixedAttr>(op.getNumeric());
  auto storage = fixed ? dyn_cast<IntegerType>(fixed.getStorage()) : IntegerType();
  if (!storage || storage.getWidth() != 16)
    return op.emitOpError("packed i16 layout requires an i16 fixed numeric policy");
  if (!llvm::all_of(types, hasSignlessI32Element))
    return op.emitOpError("packed i16 operands and results require signless i32 containers");
  if (op.getScale()->getSaturateTo() != fixed.getStorage())
    return op.emitOpError("packed i16 saturate_to must match fixed numeric storage type");
  return success();
}

static LogicalResult verifyQuantizeDomain(QuantizeOp op) {
  if (!ondrix::haveSameElementwiseShape(op.getInput().getType(), op.getResult().getType()))
    return op.emitOpError("input and result must use the same scalar or static shaped domain");
  if (ondrix::getElementTypeOrSelf(op.getInput().getType()) != getNumericStorage(op.getSrc()))
    return op.emitOpError("input element type must match source numeric storage type");
  if (ondrix::getElementTypeOrSelf(op.getResult().getType()) != getNumericStorage(op.getDst()))
    return op.emitOpError("result element type must match destination numeric storage type");
  return success();
}

static LogicalResult verifyFirWindow(FirOp op) {
  auto inputType = dyn_cast<ShapedType>(op.getInput().getType());
  auto coeffType = dyn_cast<ShapedType>(op.getCoeffs().getType());
  if (!inputType || !coeffType || !inputType.hasRank() || !coeffType.hasRank() ||
      inputType.getRank() != 1 || coeffType.getRank() != 1)
    return op.emitOpError("requires rank-1 input and coefficient windows");
  if (ondrix::isScalableVectorType(op.getInput().getType()) ||
      ondrix::isScalableVectorType(op.getCoeffs().getType()))
    return op.emitOpError("scalable vector windows are not supported");

  if (inputType.getElementType() != coeffType.getElementType())
    return op.emitOpError("input and coefficient element types must match");

  int64_t inputLength = inputType.getDimSize(0);
  int64_t coeffLength = coeffType.getDimSize(0);
  if (!ShapedType::isDynamic(inputLength) && !ShapedType::isDynamic(coeffLength) &&
      inputLength != coeffLength)
    return op.emitOpError("input and coefficient windows must have equal length");

  if (!isa<IntegerType, FloatType>(op.getResult().getType()))
    return op.emitOpError("requires a scalar integer or floating-point result");

  Type elementType = inputType.getElementType();
  if (auto fixed = dyn_cast<ondrix::ondsp::FixedAttr>(op.getNumeric())) {
    if (elementType != fixed.getStorage())
      return op.emitOpError("window element type must match fixed numeric storage type");
    auto resultType = dyn_cast<IntegerType>(op.getResult().getType());
    if (!resultType || !resultType.isSignless() || resultType.getWidth() < 32)
      return op.emitOpError("fixed FIR result must be a signless integer type of at least 32 bits");
    return success();
  }

  auto fp = cast<ondrix::ondsp::FpAttr>(op.getNumeric());
  if (elementType != fp.getFormat() || op.getResult().getType() != fp.getFormat())
    return op.emitOpError("floating-point FIR window and result types must match numeric format");
  return success();
}

static LogicalResult verifyDotDomain(DotOp op) {
  auto lhsShaped = dyn_cast<ShapedType>(op.getLhs().getType());
  auto rhsShaped = dyn_cast<ShapedType>(op.getRhs().getType());
  if (static_cast<bool>(lhsShaped) != static_cast<bool>(rhsShaped))
    return op.emitOpError("requires either two scalar operands or two rank-1 shaped operands");

  Type lhsElement = op.getLhs().getType();
  Type rhsElement = op.getRhs().getType();
  if (lhsShaped) {
    if (!lhsShaped.hasRank() || !rhsShaped.hasRank() || lhsShaped.getRank() != 1 ||
        rhsShaped.getRank() != 1)
      return op.emitOpError("shaped operands must be rank-1");
    if (ondrix::isScalableVectorType(op.getLhs().getType()) ||
        ondrix::isScalableVectorType(op.getRhs().getType()))
      return op.emitOpError("scalable vector operands are not supported");
    lhsElement = lhsShaped.getElementType();
    rhsElement = rhsShaped.getElementType();

    int64_t lhsLength = lhsShaped.getDimSize(0);
    int64_t rhsLength = rhsShaped.getDimSize(0);
    if (!ShapedType::isDynamic(lhsLength) && !ShapedType::isDynamic(rhsLength) &&
        lhsLength != rhsLength)
      return op.emitOpError("shaped operands must have equal static lengths");
  }

  if (lhsElement != rhsElement)
    return op.emitOpError("operand element types must match");

  if (auto fixed = dyn_cast<ondrix::ondsp::FixedAttr>(op.getNumeric())) {
    if (lhsElement != fixed.getStorage())
      return op.emitOpError("operand element type must match fixed numeric storage type");
    auto result = dyn_cast<IntegerType>(op.getResult().getType());
    if (!result || !result.isSignless() || result.getWidth() < 32)
      return op.emitOpError("fixed dot result must be a signless integer type of at least 32 bits");
    return success();
  }

  auto fp = cast<ondrix::ondsp::FpAttr>(op.getNumeric());
  if (lhsElement != fp.getFormat() || op.getResult().getType() != fp.getFormat())
    return op.emitOpError("floating-point dot operands and result must match numeric format");
  return success();
}

} // namespace

void FirOp::getEffects(SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  addMemRefReadEffect(getInput(), effects);
  addMemRefReadEffect(getCoeffs(), effects);
}

Speculation::Speculatability FirOp::getSpeculatability() {
  return (ondrix::requiresConservativeDSPSpeculation(getInput().getType()) ||
          ondrix::requiresConservativeDSPSpeculation(getCoeffs().getType()))
             ? Speculation::NotSpeculatable
             : Speculation::Speculatable;
}

void DotOp::getEffects(SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  addMemRefReadEffect(getLhs(), effects);
  addMemRefReadEffect(getRhs(), effects);
}

Speculation::Speculatability DotOp::getSpeculatability() {
  return (ondrix::requiresConservativeDSPSpeculation(getLhs().getType()) ||
          ondrix::requiresConservativeDSPSpeculation(getRhs().getType()))
             ? Speculation::NotSpeculatable
             : Speculation::Speculatable;
}

LogicalResult FirOp::verify() {
  if (failed(verifyOptionalProductPolicy(*this, getNumeric(), getProduct())))
    return failure();
  return verifyFirWindow(*this);
}

LogicalResult DotOp::verify() {
  if (failed(verifyOptionalProductPolicy(*this, getNumeric(), getProduct())))
    return failure();
  return verifyDotDomain(*this);
}

LogicalResult ButterflyOp::verify() {
  if (failed(verifyValueOnlyTypes(*this)))
    return failure();
  if (failed(verifyButterflyPolicies(*this, getNumeric(), getProduct(), getScale())))
    return failure();
  return verifyButterflyValueDomain(*this);
}

LogicalResult QuantizeOp::verify() {
  if (failed(verifyValueOnlyTypes(*this)))
    return failure();
  return verifyQuantizeDomain(*this);
}
