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
  switch (product.getSelection()) {
  case ProductSelection::Full:
    return "full";
  case ProductSelection::HighRaw:
    return "high_raw";
  }
  return "unsupported";
}

/// Rejects a multi-lane accumulator on a consumer that has no per-lane
/// meaning. The lane parameter defaults to one, so without this check every
/// existing "is it an accumulator?" test would silently accept W independent
/// accumulators as if they were a single one.
static LogicalResult verifySingleLaneAccumulator(Operation *op, AccType accumulator,
                                                 StringRef consumer) {
  if (isSingleLaneAccumulator(accumulator))
    return success();
  return op->emitOpError() << consumer
                           << " requires a single-lane accumulator; lanes > 1 is accepted only by "
                              "acc_zero, mac, and acc_export";
}

/// Returns the lane count of a value in the accumulator's lane domain: the
/// element count for a fixed-length rank-1 vector, and nothing for any other
/// shape. A scalar has no lane count of its own; it is the single-lane form.
static std::optional<int64_t> getVectorLaneCount(Type type) {
  auto vector = dyn_cast<VectorType>(type);
  if (!vector || vector.isScalable() || vector.getRank() != 1)
    return std::nullopt;
  return vector.getNumElements();
}

/// Verifies that a value operand or result carries exactly one element per
/// accumulator lane. A single-lane accumulator requires a scalar rather than a
/// one-element vector: the two would denote the same values but only the scalar
/// form is the one every existing consumer lowers, so admitting the vector form
/// would create a shape no lowering handles.
static LogicalResult verifyLaneDomain(Operation *op, Type type, AccType accumulator,
                                      StringRef valueName) {
  if (isSingleLaneAccumulator(accumulator)) {
    if (!isa<IntegerType>(type))
      return op->emitOpError() << valueName
                               << " must be a scalar value for a single-lane accumulator";
    return success();
  }
  std::optional<int64_t> lanes = getVectorLaneCount(type);
  if (!lanes)
    return op->emitOpError() << valueName
                             << " must be a fixed-length rank-1 vector value for a multi-lane "
                                "accumulator";
  if (*lanes != static_cast<int64_t>(accumulator.getLanes()))
    return op->emitOpError() << valueName << " lane count " << *lanes
                             << " does not match accumulator lanes " << accumulator.getLanes();
  return success();
}

static LogicalResult verifyMacLike(Operation *op, Value acc, Value lhs, Value rhs,
                                   FixedAttr numeric, ProductAttr product) {
  if (failed(verifyFixedStorageOperands(op, numeric, lhs, rhs)))
    return failure();

  auto accumulator = acc.getType().cast<AccType>();
  // The coefficient stays scalar in every lane profile: the splat is declared
  // semantics, not a lowering detail, so a vector coefficient is a different
  // operation rather than a wider form of this one.
  if (!isa<IntegerType>(rhs.getType()))
    return op->emitOpError("coefficient must be a scalar value");
  if (failed(verifyLaneDomain(op, lhs.getType(), accumulator, "value")))
    return failure();

  if (accumulator.getSignedness() != numeric.getSignedness())
    return op->emitOpError("accumulator signedness must match the fixed numeric policy");

  FailureOr<ProductSemantics> semantics = inferProductSemantics(op, numeric, product);
  if (failed(semantics))
    return failure();

  if (accumulator.getFrac() != semantics->frac)
    return op->emitOpError() << "accumulator frac " << accumulator.getFrac()
                             << " does not match expected frac " << semantics->frac << " for "
                             << getProductName(product) << " product";
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
  if (isa<RankedTensorType>(op.getLhs().getType()) || isa<RankedTensorType>(op.getRhs().getType()))
    return op.emitOpError(
        "tensor reduce_mac operands have no executable consumer; use memrefs or fixed vectors");

  auto lhs = dyn_cast<ShapedType>(op.getLhs().getType());
  auto rhs = dyn_cast<ShapedType>(op.getRhs().getType());
  if (!lhs || !rhs || !lhs.hasRank() || !rhs.hasRank() || lhs.getRank() != 1 || rhs.getRank() != 1)
    return op.emitOpError("shaped operands must be rank-1");
  if (ondrix::isScalableVectorType(op.getLhs().getType()) ||
      ondrix::isScalableVectorType(op.getRhs().getType()))
    return op.emitOpError("scalable vector operands are not supported");
  if (isa<FpAttr>(op.getNumeric()) &&
      (isa<VectorType>(op.getLhs().getType()) || isa<VectorType>(op.getRhs().getType())))
    return op.emitOpError("floating-point vector reduce_mac operands have no executable consumer");
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

LogicalResult RoundDivOp::verify() {
  if (failed(verifyValueOnlyTypes(*this)))
    return failure();
  if (failed(verifySameElementwiseShape(*this, {getInput().getType(), getResult().getType()})))
    return failure();
  if (failed(verifySignlessIntegerElementType(*this, getInput().getType(), "input")) ||
      failed(verifySignlessIntegerElementType(*this, getResult().getType(), "result")))
    return failure();
  // The generated I64Attr getters hand back unsigned values; the contract's
  // sign checks need the signed reading.
  int64_t divisor = int64_t(getDivisor());
  if (divisor < 1)
    return emitOpError("round_div requires a positive static divisor; a runtime divisor is a "
                       "different operation with an explicit zero policy");
  int64_t preShift = int64_t(getPreShiftLeft());
  if (preShift < 0 || preShift > 63)
    return emitOpError("pre_shift_left must lie in [0, 63]");
  auto inputElement = cast<IntegerType>(ondrix::getElementTypeOrSelf(getInput().getType()));
  auto resultElement = cast<IntegerType>(ondrix::getElementTypeOrSelf(getResult().getType()));
  uint64_t carrierWidth = uint64_t{inputElement.getWidth()} + uint64_t(preShift);
  if (carrierWidth > 128)
    return emitOpError("the exact scaled carrier (input width + pre_shift_left) must not "
                       "exceed 128 bits");
  // The divisor participates in carrier arithmetic (the lowering compares the
  // remainder against divisor - remainder), so it must be a representable
  // positive carrier value.
  if (carrierWidth < 64 && uint64_t(divisor) >= (uint64_t{1} << (carrierWidth - 1)))
    return emitOpError("the divisor must be representable in the exact scaled carrier");
  if (resultElement.getWidth() > carrierWidth)
    return emitOpError("round_div does not widen: the result storage must not exceed the "
                       "exact scaled carrier");
  return success();
}

LogicalResult SatCastOp::verify() {
  if (failed(verifyValueOnlyTypes(*this)))
    return failure();
  if (failed(verifySameElementwiseShape(*this, {getInput().getType(), getResult().getType()})))
    return failure();
  Type destination = getNumericStorage(getNumeric());
  return verifyResultElementType(*this, getResult().getType(), destination);
}

LogicalResult SqrtFixedOp::verify() {
  if (failed(verifyValueOnlyTypes(*this)))
    return failure();
  if (failed(verifySameElementwiseShape(*this, {getInput().getType(), getResult().getType()})))
    return failure();
  if (failed(verifySignlessIntegerElementType(*this, getInput().getType(), "input")))
    return failure();
  if (failed(verifySignlessIntegerElementType(*this, getResult().getType(), "result")))
    return failure();
  RoundingMode rounding = getRounding();
  if (rounding != RoundingMode::TowardNegative && rounding != RoundingMode::NearestEven)
    return emitOpError("sqrt_fixed supports toward_negative or nearest_even rounding");
  return success();
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

LogicalResult AddShiftOp::verify() {
  if (failed(verifyValueOnlyTypes(*this)))
    return failure();
  return verifyBinaryShiftValueDomain(*this, getLhs(), getRhs(), getResult(), getScale());
}

LogicalResult SubShiftOp::verify() {
  if (failed(verifyValueOnlyTypes(*this)))
    return failure();
  return verifyBinaryShiftValueDomain(*this, getLhs(), getRhs(), getResult(), getScale());
}

LogicalResult AccImportOp::verify() {
  FixedAttr source = getSrc();
  AccType accumulator = getAcc().getType();
  if (failed(verifySingleLaneAccumulator(*this, accumulator, "acc_import")))
    return failure();
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
  if (failed(verifySingleLaneAccumulator(*this, getAcc().getType(), "mac_sub")))
    return failure();
  return verifyMacLike(*this, getAcc(), getLhs(), getRhs(), getNumeric(), getProduct());
}

LogicalResult AccAddTermOp::verify() {
  AccType accumulator = getAcc().getType();
  FixedAttr termNumeric = getTermNumeric();
  if (failed(verifySingleLaneAccumulator(*this, accumulator, "acc_add_term")))
    return failure();
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
  if (ondrix::getElementTypeOrSelf(getResult().getType()) != destination.getStorage())
    return emitOpError("result type must match destination storage type");
  return verifyLaneDomain(*this, getResult().getType(), accumulator, "result");
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
    if (failed(verifySingleLaneAccumulator(*this, accumulator, "reduce_mac")))
      return failure();
    // `verifyMacLike` reads the operands as one value plus one scalar
    // coefficient; a reduction pairs two sequences instead, so only the policy
    // half of that check applies here.
    if (failed(verifyFixedStorageOperands(*this, fixed, getLhs(), getRhs())))
      return failure();
    if (accumulator.getSignedness() != fixed.getSignedness())
      return emitOpError("accumulator signedness must match the fixed numeric policy");
    FailureOr<ProductSemantics> semantics = inferProductSemantics(*this, fixed, *getProduct());
    if (failed(semantics))
      return failure();
    if (accumulator.getFrac() != semantics->frac)
      return emitOpError() << "accumulator frac " << accumulator.getFrac()
                           << " does not match expected frac " << semantics->frac << " for "
                           << getProductName(*getProduct()) << " product";
    return success();
  }
  if (failed(verifyValueNumericType(*this, getLhs().getType(), getNumeric(), "lhs")) ||
      failed(verifyValueNumericType(*this, getRhs().getType(), getNumeric(), "rhs")))
    return failure();
  auto fp = cast<FpAttr>(getNumeric());
  if (failed(verifyExecutableFpFormat(*this, fp, "reduce_mac")))
    return failure();
  if (getInitial().getType() != fp.getFormat())
    return emitOpError("floating-point reduce_mac initial and result must match numeric format");
  return success();
}

static LogicalResult verifyComplexValueDomain(Operation *op, TypeRange types, CxLayoutAttr layout,
                                              Attribute numeric) {
  if (failed(verifySameElementwiseShape(op, types)))
    return failure();
  // The packed Q31 layout exists only for the butterfly profile. Falling
  // through to the unpacked branch here would verify an i32 value against an
  // i32 numeric policy and silently ignore the declared packing.
  if (layout.getLayout() == ComplexLayout::PackedI32ImagHiRealLo)
    return op->emitOpError(
        "packed_i32_imag_hi_real_lo is supported only by the packed butterfly profile");
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
  // The layout selects the executable profile; container width and every
  // width-dependent numeric rule follow from it rather than being restated.
  std::optional<PackedComplexProfile> profile = getPackedComplexProfile(getLayout().getLayout());
  if (!profile)
    return emitOpError("executable butterfly requires packed_i16_imag_hi_real_lo or "
                       "packed_i32_imag_hi_real_lo layout");
  SmallVector<Type> types(getOperandTypes().begin(), getOperandTypes().end());
  types.append(getResultTypes().begin(), getResultTypes().end());
  if (failed(verifySameElementwiseShape(*this, types)))
    return failure();
  unsigned containerWidth = profile->containerWidth;
  if (!llvm::all_of(types, [containerWidth](Type type) {
        if (type.isSignlessInteger(containerWidth))
          return true;
        auto vector = dyn_cast<VectorType>(type);
        return vector && !vector.isScalable() &&
               vector.getElementType().isSignlessInteger(containerWidth);
      }))
    return emitOpError() << "executable butterfly requires scalar or fixed Vector signless i"
                         << containerWidth << " packed values";
  if (getVariant().value_or(CxButterflyVariant::Plain) != CxButterflyVariant::Plain) {
    if (profile->storageWidth != 16 || isa<VectorType>(getA().getType()))
      return emitOpError("the cross and unit combines are admitted only on scalar packed "
                         "Q15 values");
    for (ScaleAttr scale : {getProductScale(), getOutputScale()})
      if (scale.getRounding() != RoundingMode::TowardNegative &&
          scale.getRounding() != RoundingMode::NearestTiesPositive)
        return emitOpError("the cross and unit combines are admitted only under "
                           "target-inventory toward_negative or nearest_ties_positive "
                           "rounding");
  }
  return verifyPackedButterflyPolicy(*this, getLayout(), getNumeric(), getProduct(),
                                     getProductScale(), getOutputScale(),
                                     /*targetInventory=*/true);
}

LogicalResult FftStageOp::verify() { return verifyValueOnlyTypes(*this); }
