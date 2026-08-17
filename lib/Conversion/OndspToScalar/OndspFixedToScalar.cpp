#include "ondrix/Analysis/CanonicalTwiddleAnalysis.h"
#include "ondrix/Conversion/OndspToScalar/OndspToScalar.h"
#include "ondrix/Conversion/Utils/ConversionLegality.h"
#include "ondrix/Conversion/Utils/FixedPointDomainUtils.h"
#include "ondrix/Conversion/Utils/ReductionUtils.h"
#include "ondrix/Conversion/Utils/ValueTypeConversions.h"

#include "ondrix/Dialect/ondsp/IR/OndspDialect.h"
#include "ondrix/Dialect/ondsp/IR/OndspOps.h"
#include "ondrix/Dialect/ondsp/IR/OndspSemantics.h"
#include "ondrix/Dialect/ondsp/IR/OndspTypes.h"
#include "ondrix/Support/DSPTypeUtils.h"
#include "ondrix/Support/FixedPointSemantics.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/AttrTypeSubElements.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include <optional>
#include <utility>

namespace ondrix {
#define GEN_PASS_DEF_CONVERTONDSPFIXEDTOSCALAR
#include "ondrix/Conversion/Passes.h.inc"
} // namespace ondrix

using namespace mlir;

namespace {

static bool isSignedFixed(ondrix::ondsp::FixedAttr numeric, unsigned width, unsigned frac) {
  auto storage = dyn_cast<IntegerType>(numeric.getStorage());
  return storage && storage.isSignless() && storage.getWidth() == width &&
         numeric.getFrac() == frac && numeric.getSignedness() == ondrix::ondsp::Signedness::Signed;
}

static bool isSupportedAccumulator(ondrix::ondsp::AccType accumulator) {
  auto storage = dyn_cast<IntegerType>(accumulator.getStorage());
  bool isSignedFrac30 = storage && storage.isSignless() && storage.getWidth() >= 32 &&
                        accumulator.getFrac() == 30 &&
                        accumulator.getSignedness() == ondrix::ondsp::Signedness::Signed;
  return isSignedFrac30 || ondrix::ondsp::isSignedI64Frac62Accumulator(accumulator);
}

static bool isSupportedImport(ondrix::ondsp::AccType accumulator, ondrix::ondsp::FixedAttr source) {
  // Importing one scalar value into W independent lanes is not a defined
  // operation, so the lane count is part of this capability gate.
  if (!ondrix::ondsp::isSingleLaneAccumulator(accumulator))
    return false;
  if (ondrix::ondsp::isSignedI64Frac62Accumulator(accumulator))
    return ondrix::ondsp::isSignedQ31(source);
  if (!isSupportedAccumulator(accumulator) || accumulator.getFrac() != 30)
    return false;
  return ondrix::ondsp::isSignedQ15(source) || isSignedFixed(source, 32, 30);
}

static bool isSignedFixedStorage(ondrix::ondsp::FixedAttr numeric, unsigned width) {
  auto storage = dyn_cast<IntegerType>(numeric.getStorage());
  return storage && storage.isSignless() && storage.getWidth() == width &&
         numeric.getSignedness() == ondrix::ondsp::Signedness::Signed;
}

static bool isSupportedExport(ondrix::ondsp::AccType accumulator,
                              ondrix::ondsp::FixedAttr destination) {
  if (ondrix::ondsp::isSignedI64Frac62Accumulator(accumulator))
    return ondrix::ondsp::isSignedQ31(destination);
  if (!isSupportedAccumulator(accumulator) || accumulator.getFrac() != 30)
    return false;
  // Every export is a value-preserving format conversion: the destination
  // frac is the reading of the result, never a shift selector, so any i32
  // destination the verifier admits (`dst.frac <= acc.frac`) lowers through
  // the width-generic body. A boundary that changes the value belongs to
  // `round_shift`; the i64 frac-30 destination is the identity
  // materialization feeding exactly those. Every other destination is refused.
  return ondrix::ondsp::isSignedQ15(destination) || isSignedFixedStorage(destination, 32) ||
         (isSignedFixedStorage(destination, 64) && destination.getFrac() == 30);
}

static bool isSupportedAccumulatorTerm(ondrix::ondsp::AccType accumulator,
                                       ondrix::ondsp::FixedAttr numeric) {
  auto storage = dyn_cast<IntegerType>(numeric.getStorage());
  return isSupportedAccumulator(accumulator) && storage && storage.isSignless() &&
         numeric.getSignedness() == ondrix::ondsp::Signedness::Signed &&
         numeric.getFrac() == accumulator.getFrac();
}

/// Storage carrier of one accumulator value: the raw storage type for a single
/// lane, and one storage element per lane otherwise. Lanes never interact, so
/// the multi-lane carrier is the single-lane carrier applied elementwise.
static Type getAccumulatorCarrier(ondrix::ondsp::AccType accumulator) {
  if (ondrix::ondsp::isSingleLaneAccumulator(accumulator))
    return accumulator.getStorage();
  return VectorType::get({static_cast<int64_t>(accumulator.getLanes())}, accumulator.getStorage());
}

class OndspFixedToScalarTypeConverter final : public TypeConverter {
public:
  OndspFixedToScalarTypeConverter() {
    addConversion([](Type type) { return type; });
    addConversion([](ondrix::ondsp::AccType type,
                     SmallVectorImpl<Type> &results) -> std::optional<LogicalResult> {
      if (!isSupportedAccumulator(type))
        return failure();
      results.push_back(getAccumulatorCarrier(type));
      return success();
    });
  }
};

static bool isOndspType(Type type) {
  return type.getDialect().getNamespace() == ondrix::ondsp::OndspDialect::getDialectNamespace();
}

static bool isOndspAttribute(Attribute attribute) {
  return attribute.getDialect().getNamespace() ==
         ondrix::ondsp::OndspDialect::getDialectNamespace();
}

static bool containsOndspAccumulator(Type type) {
  return ondrix::conversion::containsMatchingType(
      type, [](Type nested) { return isa<ondrix::ondsp::AccType>(nested); });
}

static bool containsOndspAccumulator(TypeRange types) {
  return ondrix::conversion::containsMatchingType(
      types, [](Type nested) { return isa<ondrix::ondsp::AccType>(nested); });
}

static bool isNestedAccumulatorContainer(Type type) {
  return !isa<ondrix::ondsp::AccType>(type) && containsOndspAccumulator(type);
}

static bool containsOndspArtifact(Attribute attribute) {
  return ondrix::conversion::containsMatchingType(attribute, isOndspType) ||
         ondrix::conversion::containsMatchingAttribute(attribute, isOndspAttribute);
}

static LogicalResult verifySourceArtifactUsage(Operation *root) {
  WalkResult result = root->walk([](Operation *op) {
    auto containsNested = [](TypeRange types) {
      return llvm::any_of(types, isNestedAccumulatorContainer);
    };
    if (containsNested(op->getOperandTypes()) || containsNested(op->getResultTypes())) {
      op->emitOpError("nested accumulator containers are unsupported");
      return WalkResult::interrupt();
    }
    if (auto function = dyn_cast<func::FuncOp>(op)) {
      FunctionType type = function.getFunctionType();
      if (containsNested(type.getInputs()) || containsNested(type.getResults())) {
        op->emitOpError("nested accumulator containers are unsupported");
        return WalkResult::interrupt();
      }
    }
    for (Region &region : op->getRegions())
      for (Block &block : region)
        if (containsNested(block.getArgumentTypes())) {
          op->emitOpError("nested accumulator containers are unsupported");
          return WalkResult::interrupt();
        }

    auto function = dyn_cast<func::FuncOp>(op);
    for (NamedAttribute namedAttribute : op->getAttrs()) {
      // Function signature types are converted by the standard function
      // conversion patterns and were checked structurally above.
      if (function && namedAttribute.getName() == function.getFunctionTypeAttrName())
        continue;
      // Attributes owned by source operations are consumed with those
      // operations. Host-operation metadata has no generic conversion.
      if (op->getDialect() &&
          op->getDialect()->getNamespace() == ondrix::ondsp::OndspDialect::getDialectNamespace())
        continue;
      if (!containsOndspArtifact(namedAttribute.getValue()))
        continue;
      op->emitOpError() << "attribute '" << namedAttribute.getName().getValue()
                        << "' contains an Ondsp type or attribute; source artifacts in host "
                           "metadata are unsupported";
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return failure(result.wasInterrupted());
}

static bool hasLegalConvertedTypes(Operation *op, TypeConverter &typeConverter) {
  return ondrix::conversion::hasLegalConvertedTypesAndAttributes(op, typeConverter, isOndspType,
                                                                 isOndspAttribute);
}

// The arithmetic below is written once and applied either to a scalar carrier
// or elementwise to a `vector<lanes x carrier>`. Every helper therefore takes
// the carrier type from its input instead of assuming a scalar, so a multi-lane
// accumulator executes the identical per-lane sequence.
static IntegerType getIntegerElementType(Type type) {
  if (auto vector = dyn_cast<VectorType>(type))
    return cast<IntegerType>(vector.getElementType());
  return cast<IntegerType>(type);
}

static Type getIntegerTypeLike(Type reference, unsigned width, OpBuilder &builder) {
  IntegerType elementType = builder.getIntegerType(width);
  if (auto vector = dyn_cast<VectorType>(reference))
    return VectorType::get(vector.getShape(), elementType);
  return elementType;
}

static Value createIntegerConstant(Location loc, Type type, const llvm::APInt &value,
                                   OpBuilder &builder) {
  IntegerType elementType = getIntegerElementType(type);
  IntegerAttr valueAttr = builder.getIntegerAttr(elementType, value);
  if (auto vector = dyn_cast<VectorType>(type))
    return builder.create<arith::ConstantOp>(loc, vector,
                                             SplatElementsAttr::get(vector, valueAttr));
  return builder.create<arith::ConstantOp>(loc, type, valueAttr);
}

static Value createIntegerConstant(Location loc, Type type, int64_t value, OpBuilder &builder) {
  return createIntegerConstant(
      loc, type, llvm::APInt(getIntegerElementType(type).getWidth(), value, true), builder);
}

class AccZeroOpLowering final : public OpConversionPattern<ondrix::ondsp::AccZeroOp> {
public:
  using OpConversionPattern<ondrix::ondsp::AccZeroOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(ondrix::ondsp::AccZeroOp op, OpAdaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Type resultType = getTypeConverter()->convertType(op.getAcc().getType());
    if (!resultType || !isa<IntegerType>(ondrix::getElementTypeOrSelf(resultType)))
      return op.emitOpError("fixed scalar lowering requires a supported accumulator type");
    rewriter.replaceOp(op, createIntegerConstant(op.getLoc(), resultType, 0, rewriter));
    return success();
  }
};

class AccImportOpLowering final : public OpConversionPattern<ondrix::ondsp::AccImportOp> {
public:
  using OpConversionPattern<ondrix::ondsp::AccImportOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(ondrix::ondsp::AccImportOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto accumulator = cast<ondrix::ondsp::AccType>(op.getAcc().getType());
    if (!isSupportedImport(accumulator, op.getSrc()))
      return op.emitOpError(
          "fixed scalar lowering supports Q15 or Q30 to a signed frac30 accumulator of at least "
          "32 bits, or Q31 to i64/frac62 exact import");

    auto accumulatorStorage = cast<IntegerType>(accumulator.getStorage());
    Value extended =
        rewriter.create<arith::ExtSIOp>(op.getLoc(), accumulatorStorage, adaptor.getInput());
    Value shift = rewriter.create<arith::ConstantIntOp>(
        op.getLoc(), accumulator.getFrac() - op.getSrc().getFrac(), accumulatorStorage.getWidth());
    rewriter.replaceOpWithNewOp<arith::ShLIOp>(op, extended, shift);
    return success();
  }
};

static Value lowerAccumulatorUpdate(Location loc, Value accumulator, Value product,
                                    ondrix::ondsp::OverflowMode overflowMode,
                                    ondrix::fixedpoint::AccumulatorUpdateOperation operation,
                                    OpBuilder &builder) {
  Type accumulatorType = accumulator.getType();
  IntegerType accumulatorElement = getIntegerElementType(accumulatorType);
  IntegerType productElement = getIntegerElementType(product.getType());
  unsigned intermediateWidth = ondrix::fixedpoint::getAccumulatorUpdateIntermediateWidth(
      accumulatorElement.getWidth(), productElement.getWidth());
  Type intermediateType = getIntegerTypeLike(accumulatorType, intermediateWidth, builder);
  Value extendedAccumulator = builder.create<arith::ExtSIOp>(loc, intermediateType, accumulator);
  Value extendedProduct = builder.create<arith::ExtSIOp>(loc, intermediateType, product);
  Value updated;
  switch (operation) {
  case ondrix::fixedpoint::AccumulatorUpdateOperation::Add:
    updated = builder.create<arith::AddIOp>(loc, extendedAccumulator, extendedProduct);
    break;
  case ondrix::fixedpoint::AccumulatorUpdateOperation::Subtract:
    updated = builder.create<arith::SubIOp>(loc, extendedAccumulator, extendedProduct);
    break;
  }

  if (overflowMode == ondrix::ondsp::OverflowMode::Wrap)
    return builder.create<arith::TruncIOp>(loc, accumulatorType, updated);

  // Clamp in the exact update width before narrowing to the accumulator. The
  // comparison and select vectorize elementwise, so every lane saturates
  // independently against the same accumulator bounds.
  llvm::APInt minimum =
      llvm::APInt::getSignedMinValue(accumulatorElement.getWidth()).sext(intermediateWidth);
  llvm::APInt maximum =
      llvm::APInt::getSignedMaxValue(accumulatorElement.getWidth()).sext(intermediateWidth);
  Value minimumValue = createIntegerConstant(loc, intermediateType, minimum, builder);
  Value maximumValue = createIntegerConstant(loc, intermediateType, maximum, builder);
  Value belowMinimum =
      builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, updated, minimumValue);
  Value aboveMaximum =
      builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sgt, updated, maximumValue);
  Value lowerClamped = builder.create<arith::SelectOp>(loc, belowMinimum, minimumValue, updated);
  Value clamped = builder.create<arith::SelectOp>(loc, aboveMaximum, maximumValue, lowerClamped);
  return builder.create<arith::TruncIOp>(loc, accumulatorType, clamped);
}

static Value roundSignedRightShift(Location loc, Value input, unsigned shift,
                                   ondrix::ondsp::RoundingMode roundingMode,
                                   ConversionPatternRewriter &rewriter) {
  Type type = input.getType();
  if (shift == 0)
    return input;

  Value shiftValue = createIntegerConstant(loc, type, shift, rewriter);
  Value quotient = rewriter.create<arith::ShRSIOp>(loc, input, shiftValue);
  if (roundingMode == ondrix::ondsp::RoundingMode::TowardNegative)
    return quotient;

  Type remainderBitsType = getIntegerTypeLike(type, shift, rewriter);
  Value remainderBits = rewriter.create<arith::TruncIOp>(loc, remainderBitsType, input);
  Value remainder = rewriter.create<arith::ExtUIOp>(loc, type, remainderBits);
  Value zero = createIntegerConstant(loc, type, 0, rewriter);
  Value one = createIntegerConstant(loc, type, 1, rewriter);
  Value incrementCondition;

  switch (roundingMode) {
  case ondrix::ondsp::RoundingMode::TowardNegative:
    llvm_unreachable("toward-negative rounding returned above");
  case ondrix::ondsp::RoundingMode::TowardZero: {
    Value isNegative = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, input, zero);
    Value hasRemainder =
        rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ne, remainder, zero);
    incrementCondition = rewriter.create<arith::AndIOp>(loc, isNegative, hasRemainder);
    break;
  }
  case ondrix::ondsp::RoundingMode::NearestTiesPositive: {
    // Ties toward +infinity. The textbook add-half-then-shift form is
    // deliberately NOT used: adding 2^(shift-1) in the input width overflows
    // near the maximum representable value, which would silently change the
    // result of exactly the inputs a saturating boundary cares about. The
    // quotient/remainder form is total: floor already happened, so a
    // remainder of at least half moves the result one step up.
    Value half = createIntegerConstant(loc, type, int64_t{1} << (shift - 1), rewriter);
    incrementCondition =
        rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::uge, remainder, half);
    break;
  }
  case ondrix::ondsp::RoundingMode::NearestEven: {
    Value half = createIntegerConstant(loc, type, int64_t{1} << (shift - 1), rewriter);
    Value aboveHalf =
        rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ugt, remainder, half);
    Value exactlyHalf =
        rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, remainder, half);
    Value quotientLowBit = rewriter.create<arith::AndIOp>(loc, quotient, one);
    Value quotientIsOdd =
        rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ne, quotientLowBit, zero);
    Value halfAndOdd = rewriter.create<arith::AndIOp>(loc, exactlyHalf, quotientIsOdd);
    incrementCondition = rewriter.create<arith::OrIOp>(loc, aboveHalf, halfAndOdd);
    break;
  }
  }

  Value increment = rewriter.create<arith::SelectOp>(loc, incrementCondition, one, zero);
  return rewriter.create<arith::AddIOp>(loc, quotient, increment);
}

static Value narrowSignedValue(Location loc, Value input, Type destinationType,
                               ondrix::ondsp::OverflowMode overflowMode,
                               ConversionPatternRewriter &rewriter) {
  Type inputType = input.getType();
  if (inputType == destinationType)
    return input;
  if (overflowMode == ondrix::ondsp::OverflowMode::Wrap)
    return rewriter.create<arith::TruncIOp>(loc, destinationType, input);

  IntegerType inputElementType = getIntegerElementType(inputType);
  IntegerType destinationElementType = getIntegerElementType(destinationType);
  llvm::APInt minimum = llvm::APInt::getSignedMinValue(destinationElementType.getWidth())
                            .sext(inputElementType.getWidth());
  llvm::APInt maximum = llvm::APInt::getSignedMaxValue(destinationElementType.getWidth())
                            .sext(inputElementType.getWidth());
  Value minimumValue = createIntegerConstant(loc, inputType, minimum, rewriter);
  Value maximumValue = createIntegerConstant(loc, inputType, maximum, rewriter);
  Value belowMinimum =
      rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, input, minimumValue);
  Value aboveMaximum =
      rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sgt, input, maximumValue);
  Value lowerClamped = rewriter.create<arith::SelectOp>(loc, belowMinimum, minimumValue, input);
  Value clamped = rewriter.create<arith::SelectOp>(loc, aboveMaximum, maximumValue, lowerClamped);
  return rewriter.create<arith::TruncIOp>(loc, destinationType, clamped);
}

static Value requantizeSignedValue(Location loc, Value input, ondrix::ondsp::ScaleAttr scale,
                                   ConversionPatternRewriter &rewriter) {
  Type inputType = input.getType();
  Value shifted = input;
  if (scale.getPreShiftLeft() != 0) {
    Value amount = createIntegerConstant(loc, inputType, scale.getPreShiftLeft(), rewriter);
    shifted = rewriter.create<arith::ShLIOp>(loc, input, amount);
  }
  Value rounded =
      roundSignedRightShift(loc, shifted, scale.getPostShiftRight(), scale.getRounding(), rewriter);
  unsigned destinationWidth = cast<IntegerType>(scale.getSaturateTo()).getWidth();
  return narrowSignedValue(loc, rounded, getIntegerTypeLike(inputType, destinationWidth, rewriter),
                           scale.getOverflow(), rewriter);
}

// Split one packed container into its signed components. The container is
// exactly two components wide, so the real part is the low truncation and the
// imaginary part is the logical high half.
static std::pair<Value, Value> unpackPackedComplex(Location loc, Value packed,
                                                   unsigned storageWidth,
                                                   ConversionPatternRewriter &rewriter) {
  Type component = getIntegerTypeLike(packed.getType(), storageWidth, rewriter);
  Type container = packed.getType();
  Value real = rewriter.create<arith::TruncIOp>(loc, component, packed);
  Value shift = createIntegerConstant(loc, container, storageWidth, rewriter);
  Value high = rewriter.create<arith::ShRUIOp>(loc, packed, shift);
  Value imaginary = rewriter.create<arith::TruncIOp>(loc, component, high);
  return {real, imaginary};
}

static Value packComplex(Location loc, Value real, Value imaginary, unsigned storageWidth,
                         ConversionPatternRewriter &rewriter) {
  Type container = getIntegerTypeLike(real.getType(), 2 * storageWidth, rewriter);
  Value realBits = rewriter.create<arith::ExtUIOp>(loc, container, real);
  Value imaginaryBits = rewriter.create<arith::ExtUIOp>(loc, container, imaginary);
  Value shift = createIntegerConstant(loc, container, storageWidth, rewriter);
  Value shiftedImaginary = rewriter.create<arith::ShLIOp>(loc, imaginaryBits, shift);
  return rewriter.create<arith::OrIOp>(loc, shiftedImaginary, realBits);
}

// Exact form of nearest-even saturating Q15 multiplication by 32767/32768.
static Value multiplyByPackedQ15One(Location loc, Value input,
                                    ConversionPatternRewriter &rewriter) {
  IntegerType i16 = rewriter.getI16Type();
  Value negativeThreshold = createIntegerConstant(loc, i16, -16384, rewriter);
  Value positiveThreshold = createIntegerConstant(loc, i16, 16384, rewriter);
  Value one = createIntegerConstant(loc, i16, 1, rewriter);
  Value zero = createIntegerConstant(loc, i16, 0, rewriter);
  Value below =
      rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, input, negativeThreshold);
  Value above =
      rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sgt, input, positiveThreshold);
  Value increment = rewriter.create<arith::SelectOp>(loc, below, one, zero);
  Value decrement = rewriter.create<arith::SelectOp>(loc, above, one, zero);
  Value adjusted = rewriter.create<arith::AddIOp>(loc, input, increment);
  return rewriter.create<arith::SubIOp>(loc, adjusted, decrement);
}

static Value saturatingNegatePackedQ15(Location loc, Value input,
                                       ConversionPatternRewriter &rewriter) {
  IntegerType i16 = rewriter.getI16Type();
  Value zero = createIntegerConstant(loc, i16, 0, rewriter);
  Value minimum = createIntegerConstant(loc, i16, -32768, rewriter);
  Value maximum = createIntegerConstant(loc, i16, 32767, rewriter);
  Value isMinimum = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, input, minimum);
  Value negated = rewriter.create<arith::SubIOp>(loc, zero, input);
  return rewriter.create<arith::SelectOp>(loc, isMinimum, maximum, negated);
}

static Value lowerSignedProduct(Location loc, Value lhs, Value rhs,
                                ondrix::ondsp::FixedAttr numeric,
                                ondrix::ondsp::ProductSemantics semantics, OpBuilder &builder) {
  unsigned storageWidth = cast<IntegerType>(numeric.getStorage()).getWidth();
  Type fullProductType = getIntegerTypeLike(lhs.getType(), storageWidth * 2, builder);
  Value lhsExtended = builder.create<arith::ExtSIOp>(loc, fullProductType, lhs);
  Value rhsExtended = builder.create<arith::ExtSIOp>(loc, fullProductType, rhs);
  Value fullProduct = builder.create<arith::MulIOp>(loc, lhsExtended, rhsExtended);
  if (semantics.selection == ondrix::ondsp::ProductSelection::Full)
    return fullProduct;

  Value shift = createIntegerConstant(loc, fullProductType, storageWidth, builder);
  Value shifted = builder.create<arith::ShRSIOp>(loc, fullProduct, shift);
  Type rawHighType = getIntegerTypeLike(lhs.getType(), semantics.rawWidth, builder);
  return builder.create<arith::TruncIOp>(loc, rawHighType, shifted);
}

template <typename OpTy, ondrix::fixedpoint::AccumulatorUpdateOperation operation>
class MacLikeOpLowering final : public OpConversionPattern<OpTy> {
public:
  using OpConversionPattern<OpTy>::OpConversionPattern;

  LogicalResult matchAndRewrite(OpTy op, typename OpTy::Adaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto accumulator = cast<ondrix::ondsp::AccType>(op.getAcc().getType());
    // `mac` is the only multi-lane accumulator update. Every other MAC-like
    // operation fails closed here as well as in its verifier, so a pass that
    // builds one without going through the parser cannot slip past.
    if (!std::is_same_v<OpTy, ondrix::ondsp::MacOp> &&
        !ondrix::ondsp::isSingleLaneAccumulator(accumulator))
      return op.emitOpError("fixed scalar lowering supports multi-lane accumulators only for mac");
    FailureOr<ondrix::conversion::SupportedFixedMacDomain> domain =
        ondrix::conversion::getSupportedFixedScalarMacDomain(op, accumulator, op.getNumeric(),
                                                             op.getProduct());
    if (failed(domain))
      return op.emitOpError(
          "fixed scalar lowering supports Q15/full with a signed frac30 accumulator of at least "
          "32 bits, Q31/full with i64/frac62, or Q31/high_raw with i40/frac30 accumulation");

    Location loc = op.getLoc();
    Value value = adaptor.getLhs();
    Value coefficient = adaptor.getRhs();
    if (!ondrix::ondsp::isSingleLaneAccumulator(accumulator)) {
      auto valueType = dyn_cast<VectorType>(value.getType());
      if (!valueType || valueType.isScalable() || valueType.getRank() != 1 ||
          valueType.getNumElements() != static_cast<int64_t>(accumulator.getLanes()))
        return op.emitOpError("fixed scalar lowering requires one value lane per accumulator lane");
      // The declared broadcast of the scalar coefficient across the lanes. It
      // is materialized here rather than assumed by the caller because it is
      // part of the operation's meaning.
      coefficient = rewriter.create<vector::BroadcastOp>(
          loc, VectorType::get(valueType.getShape(), coefficient.getType()), coefficient);
    }

    Value product =
        lowerSignedProduct(loc, value, coefficient, op.getNumeric(), domain->product, rewriter);
    Value updated = lowerAccumulatorUpdate(loc, adaptor.getAcc(), product,
                                           accumulator.getUpdateOverflow(), operation, rewriter);
    rewriter.replaceOp(op, updated);
    return success();
  }
};

using MacOpLowering =
    MacLikeOpLowering<ondrix::ondsp::MacOp, ondrix::fixedpoint::AccumulatorUpdateOperation::Add>;
using MacSubOpLowering =
    MacLikeOpLowering<ondrix::ondsp::MacSubOp,
                      ondrix::fixedpoint::AccumulatorUpdateOperation::Subtract>;

class CxButterflyOpLowering final : public OpConversionPattern<ondrix::ondsp::CxButterflyOp> {
public:
  CxButterflyOpLowering(TypeConverter &typeConverter, MLIRContext *context,
                        bool specializeCanonicalTwiddles)
      : OpConversionPattern(typeConverter, context),
        specializeCanonicalTwiddles(specializeCanonicalTwiddles) {}

  LogicalResult matchAndRewrite(ondrix::ondsp::CxButterflyOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    std::optional<ondrix::ondsp::PackedComplexProfile> profile =
        ondrix::ondsp::getPackedComplexProfile(op.getLayout().getLayout());
    if (!profile)
      return op.emitOpError("fixed scalar lowering requires an executable packed complex layout");
    unsigned storageWidth = profile->storageWidth;
    auto [aReal, aImaginary] = unpackPackedComplex(loc, adaptor.getA(), storageWidth, rewriter);
    auto [bReal, bImaginary] = unpackPackedComplex(loc, adaptor.getB(), storageWidth, rewriter);
    auto [wReal, wImaginary] =
        unpackPackedComplex(loc, adaptor.getTwiddle(), storageWidth, rewriter);

    Value twiddledReal;
    Value twiddledImaginary;
    bool specialized = false;
    // Canonical-twiddle specialization is proven only for packed Q15: its
    // identities are stated over i16 components and its exhaustive ground
    // truth covers that domain alone. The Q31 profile keeps the general
    // product path rather than reusing an unproven identity.
    bool cross = op.getVariant() == ondrix::ondsp::CxButterflyVariant::Cross;
    // The canonical-twiddle identities are stated over the plain combine.
    if (specializeCanonicalTwiddles && storageWidth == 16 && !cross) {
      std::optional<ondrix::analysis::CanonicalPackedQ15TwiddlePlan> plan =
          ondrix::analysis::planCanonicalPackedQ15Twiddle(op);
      if (plan) {
        if (failed(std::move(*plan).consumeIfValid(
                op, [&](ondrix::analysis::CanonicalPackedQ15TwiddleIdentity identity) {
                  switch (identity) {
                  case ondrix::analysis::CanonicalPackedQ15TwiddleIdentity::One:
                    twiddledReal = multiplyByPackedQ15One(loc, bReal, rewriter);
                    twiddledImaginary = multiplyByPackedQ15One(loc, bImaginary, rewriter);
                    break;
                  case ondrix::analysis::CanonicalPackedQ15TwiddleIdentity::MinusJ:
                    twiddledReal = bImaginary;
                    twiddledImaginary = saturatingNegatePackedQ15(loc, bReal, rewriter);
                    break;
                  }
                  specialized = true;
                  return success();
                })))
          return rewriter.notifyMatchFailure(op, "canonical twiddle authorization became stale");
      }
    }
    if (!specialized) {
      // Exact carrier for both cross sums: the binding imaginary term reaches
      // 2^63, so Q15 needs 33 bits and Q31 needs 65. i128 is the generic
      // choice the backend chain already handles, not a minimality claim.
      Type productType =
          getIntegerTypeLike(bReal.getType(), storageWidth == 16 ? 33 : 128, rewriter);
      auto extendProductOperand = [&](Value value) {
        return rewriter.create<arith::ExtSIOp>(loc, productType, value);
      };
      Value br = extendProductOperand(bReal);
      Value bi = extendProductOperand(bImaginary);
      Value wr = extendProductOperand(wReal);
      Value wi = extendProductOperand(wImaginary);
      Value brwr = rewriter.create<arith::MulIOp>(loc, br, wr);
      Value biwi = rewriter.create<arith::MulIOp>(loc, bi, wi);
      Value brwi = rewriter.create<arith::MulIOp>(loc, br, wi);
      Value biwr = rewriter.create<arith::MulIOp>(loc, bi, wr);
      Value productReal = rewriter.create<arith::SubIOp>(loc, brwr, biwi);
      Value productImaginary = rewriter.create<arith::AddIOp>(loc, brwi, biwr);
      twiddledReal = requantizeSignedValue(loc, productReal, op.getProductScale(), rewriter);
      twiddledImaginary =
          requantizeSignedValue(loc, productImaginary, op.getProductScale(), rewriter);
    }

    // Exact carrier for a +- t, one bit wider than the component storage.
    Type sumType = getIntegerTypeLike(aReal.getType(), storageWidth + 1, rewriter);
    auto extendSumOperand = [&](Value value) {
      return rewriter.create<arith::ExtSIOp>(loc, sumType, value);
    };
    Value ar = extendSumOperand(aReal);
    Value ai = extendSumOperand(aImaginary);
    Value tr = extendSumOperand(twiddledReal);
    Value ti = extendSumOperand(twiddledImaginary);
    // The cross combine forms a -+ j*t from the same requantized product.
    Value out0Real = rewriter.create<arith::AddIOp>(loc, ar, cross ? ti : tr);
    Value out0Imaginary = cross ? rewriter.create<arith::SubIOp>(loc, ai, tr).getResult()
                                : rewriter.create<arith::AddIOp>(loc, ai, ti).getResult();
    Value out1Real = rewriter.create<arith::SubIOp>(loc, ar, cross ? ti : tr);
    Value out1Imaginary = cross ? rewriter.create<arith::AddIOp>(loc, ai, tr).getResult()
                                : rewriter.create<arith::SubIOp>(loc, ai, ti).getResult();

    out0Real = requantizeSignedValue(loc, out0Real, op.getOutputScale(), rewriter);
    out0Imaginary = requantizeSignedValue(loc, out0Imaginary, op.getOutputScale(), rewriter);
    out1Real = requantizeSignedValue(loc, out1Real, op.getOutputScale(), rewriter);
    out1Imaginary = requantizeSignedValue(loc, out1Imaginary, op.getOutputScale(), rewriter);

    rewriter.replaceOp(
        op, ValueRange{packComplex(loc, out0Real, out0Imaginary, storageWidth, rewriter),
                       packComplex(loc, out1Real, out1Imaginary, storageWidth, rewriter)});
    return success();
  }

private:
  bool specializeCanonicalTwiddles;
};

class AccAddTermOpLowering final : public OpConversionPattern<ondrix::ondsp::AccAddTermOp> {
public:
  using OpConversionPattern<ondrix::ondsp::AccAddTermOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(ondrix::ondsp::AccAddTermOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto accumulator = cast<ondrix::ondsp::AccType>(op.getAcc().getType());
    if (!ondrix::ondsp::isSingleLaneAccumulator(accumulator))
      return op.emitOpError("fixed scalar lowering requires a single-lane accumulator for "
                            "acc_add_term");
    if (!isSupportedAccumulatorTerm(accumulator, op.getTermNumeric()))
      return op.emitOpError(
          "fixed scalar lowering requires a supported signed accumulator and a signed term "
          "with the same fractional position");

    Value result = lowerAccumulatorUpdate(
        op.getLoc(), adaptor.getAcc(), adaptor.getTerm(), accumulator.getUpdateOverflow(),
        ondrix::fixedpoint::AccumulatorUpdateOperation::Add, rewriter);
    rewriter.replaceOp(op, result);
    return success();
  }
};

class ReduceMacOpLowering final : public OpConversionPattern<ondrix::ondsp::ReduceMacOp> {
public:
  using OpConversionPattern<ondrix::ondsp::ReduceMacOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(ondrix::ondsp::ReduceMacOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto accumulator = dyn_cast<ondrix::ondsp::AccType>(op.getInitial().getType());
    auto numeric = dyn_cast<ondrix::ondsp::FixedAttr>(op.getNumeric());
    if (!accumulator || !numeric || !op.getProduct())
      return op.emitOpError(
          "fixed scalar reduction supports Q15/full with a signed frac30 accumulator of at least "
          "32 bits, Q31/full with i64/frac62, or Q31/high_raw with i40/frac30 accumulation");
    if (!ondrix::ondsp::isSingleLaneAccumulator(accumulator))
      return op.emitOpError("fixed scalar reduction requires a single-lane accumulator; a "
                            "reduction spends its lanes on the reduction axis");

    FailureOr<ondrix::conversion::SupportedFixedMacDomain> domain =
        ondrix::conversion::getSupportedFixedScalarMacDomain(op, accumulator, numeric,
                                                             *op.getProduct());
    if (failed(domain))
      return op.emitOpError(
          "fixed scalar reduction supports Q15/full with a signed frac30 accumulator of at least "
          "32 bits, Q31/full with i64/frac62, or Q31/high_raw with i40/frac30 accumulation");

    FailureOr<ondrix::conversion::RankOneReductionBounds> bounds =
        ondrix::conversion::createRankOneMemRefReductionBounds(
            op, adaptor.getLhs(), adaptor.getRhs(), numeric.getStorage(), "fixed scalar lowering",
            rewriter);
    if (failed(bounds))
      return failure();

    Location loc = op.getLoc();
    Value step = rewriter.create<arith::ConstantIndexOp>(loc, 1);

    auto loop = rewriter.create<scf::ForOp>(
        loc, bounds->lowerBound, bounds->upperBound, step, ValueRange{adaptor.getInitial()},
        [&](OpBuilder &builder, Location bodyLoc, Value iv, ValueRange iterArgs) {
          Value lhs = builder.create<memref::LoadOp>(bodyLoc, adaptor.getLhs(), iv);
          Value rhs = builder.create<memref::LoadOp>(bodyLoc, adaptor.getRhs(), iv);
          Value product = lowerSignedProduct(bodyLoc, lhs, rhs, numeric, domain->product, builder);
          Value next = lowerAccumulatorUpdate(
              bodyLoc, iterArgs.front(), product, accumulator.getUpdateOverflow(),
              ondrix::fixedpoint::AccumulatorUpdateOperation::Add, builder);
          builder.create<scf::YieldOp>(bodyLoc, next);
        });

    rewriter.replaceOp(op, loop.getResult(0));
    return success();
  }
};

class AccExportOpLowering final : public OpConversionPattern<ondrix::ondsp::AccExportOp> {
public:
  using OpConversionPattern<ondrix::ondsp::AccExportOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(ondrix::ondsp::AccExportOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto accumulator = cast<ondrix::ondsp::AccType>(op.getAcc().getType());
    if (!isSupportedExport(accumulator, op.getDst()))
      return op.emitOpError(
          "fixed scalar lowering supports a signed frac30 accumulator of at least 32 bits to a "
          "signed Q15, signed i32, or identity signed i64/frac30 destination, and i64/frac62 to "
          "Q31 export");

    // The op verifier already rejects destinations whose frac exceeds the
    // accumulator frac, but this lowering must not turn an unverified
    // pass-created op into an undefined shift; keep the same self-guard the
    // round_shift lowering carries.
    if (op.getDst().getFrac() > accumulator.getFrac())
      return op.emitOpError(
          "fixed scalar lowering requires the destination frac not to exceed the accumulator "
          "frac");

    unsigned shift = accumulator.getFrac() - op.getDst().getFrac();
    Value rounded =
        roundSignedRightShift(op.getLoc(), adaptor.getAcc(), shift, op.getRounding(), rewriter);
    // A multi-lane accumulator exports one destination element per lane; the
    // rounding, narrowing, and clamping sequence below is exactly the
    // single-lane one applied elementwise.
    unsigned destinationWidth = cast<IntegerType>(op.getDst().getStorage()).getWidth();
    Type destinationType = getIntegerTypeLike(rounded.getType(), destinationWidth, rewriter);
    unsigned roundedWidth = getIntegerElementType(rounded.getType()).getWidth();
    Value result;
    if (destinationWidth > roundedWidth) {
      // The destination is WIDER than the accumulator storage, which the
      // i64/frac30 identity destination reaches from an i40 or i48
      // accumulator. `narrowSignedValue` cannot serve this direction: its
      // wrap branch would emit an illegal widening `arith.trunci` and its
      // saturate branch would sign extend the destination bounds into a
      // narrower comparison width. Widening sign extension is exactly
      // value preserving, so both destination export overflow modes are
      // provably no-ops here and neither needs to be materialized. The
      // accumulator's own update_overflow stays observable semantics; the
      // widening merely materializes the already-updated state faithfully.
      result = rewriter.create<arith::ExtSIOp>(op.getLoc(), destinationType, rounded);
    } else {
      result = narrowSignedValue(op.getLoc(), rounded, destinationType, op.getOverflow(), rewriter);
    }
    rewriter.replaceOp(op, result);
    return success();
  }
};

static IntegerType getSignlessIntegerElementOrNull(Type type) {
  if (auto vector = dyn_cast<VectorType>(type)) {
    if (vector.isScalable())
      return nullptr;
    type = vector.getElementType();
  }
  auto element = dyn_cast<IntegerType>(type);
  if (!element || !element.isSignless())
    return nullptr;
  return element;
}

class RoundShiftOpLowering final : public OpConversionPattern<ondrix::ondsp::RoundShiftOp> {
public:
  using OpConversionPattern<ondrix::ondsp::RoundShiftOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(ondrix::ondsp::RoundShiftOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    IntegerType inputElement = getSignlessIntegerElementOrNull(adaptor.getInput().getType());
    if (!inputElement)
      return op.emitOpError("fixed scalar lowering supports scalar or fixed-width vector "
                            "signless integer round_shift values");
    // The registered lowering implements only the proven subset: no
    // pre-shift, an in-range shift amount, and same-or-narrower destination
    // storage. Widening or overflowing forms fail closed until a real
    // consumer defines their exact semantics.
    ondrix::ondsp::ScaleAttr scale = op.getScale();
    if (scale.getPreShiftLeft() != 0)
      return op.emitOpError("fixed scalar lowering requires round_shift pre_shift_left = 0");
    if (scale.getPostShiftRight() >= inputElement.getWidth())
      return op.emitOpError(
          "fixed scalar lowering requires round_shift post_shift_right narrower than the input "
          "storage");
    if (cast<IntegerType>(scale.getSaturateTo()).getWidth() > inputElement.getWidth())
      return op.emitOpError("fixed scalar lowering does not widen round_shift results");
    rewriter.replaceOp(
        op, requantizeSignedValue(op.getLoc(), adaptor.getInput(), op.getScale(), rewriter));
    return success();
  }
};

// The exact sum or difference of two W-bit values needs W+1 bits, so the
// carrier is widened before the scale runs; nothing is lost before the
// operation's single declared boundary.
template <typename ShiftOp, typename ArithOp>
class BinaryShiftOpLowering final : public OpConversionPattern<ShiftOp> {
public:
  using OpConversionPattern<ShiftOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(ShiftOp op, typename ShiftOp::Adaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    IntegerType inputElement = getSignlessIntegerElementOrNull(adaptor.getLhs().getType());
    if (!inputElement)
      return op.emitOpError("fixed scalar lowering supports scalar or fixed-width vector "
                            "signless integer operands");
    ondrix::ondsp::ScaleAttr scale = op.getScale();
    if (scale.getPreShiftLeft() != 0)
      return op.emitOpError("fixed scalar lowering requires pre_shift_left = 0");
    unsigned carrierWidth = inputElement.getWidth() + 1;
    if (scale.getPostShiftRight() >= carrierWidth)
      return op.emitOpError("fixed scalar lowering requires post_shift_right narrower than the "
                            "exact carrier");
    if (cast<IntegerType>(scale.getSaturateTo()).getWidth() > carrierWidth)
      return op.emitOpError("fixed scalar lowering does not widen past the exact carrier");

    Location loc = op.getLoc();
    Type carrierType = getIntegerTypeLike(adaptor.getLhs().getType(), carrierWidth, rewriter);
    Value lhs = rewriter.create<arith::ExtSIOp>(loc, carrierType, adaptor.getLhs());
    Value rhs = rewriter.create<arith::ExtSIOp>(loc, carrierType, adaptor.getRhs());
    Value exact = rewriter.create<ArithOp>(loc, lhs, rhs);
    rewriter.replaceOp(op, requantizeSignedValue(loc, exact, scale, rewriter));
    return success();
  }
};

using AddShiftOpLowering = BinaryShiftOpLowering<ondrix::ondsp::AddShiftOp, arith::AddIOp>;
using SubShiftOpLowering = BinaryShiftOpLowering<ondrix::ondsp::SubShiftOp, arith::SubIOp>;

class RoundDivOpLowering final : public OpConversionPattern<ondrix::ondsp::RoundDivOp> {
public:
  using OpConversionPattern<ondrix::ondsp::RoundDivOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(ondrix::ondsp::RoundDivOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    IntegerType inputElement = getSignlessIntegerElementOrNull(adaptor.getInput().getType());
    if (!inputElement)
      return op.emitOpError("fixed scalar lowering supports scalar or fixed-width vector "
                            "signless integer round_div values");
    Location loc = op.getLoc();
    unsigned carrierWidth = inputElement.getWidth() + unsigned(op.getPreShiftLeft());
    Type carrierType = getIntegerTypeLike(adaptor.getInput().getType(), carrierWidth, rewriter);
    Value scaled = adaptor.getInput();
    if (scaled.getType() != carrierType)
      scaled = rewriter.create<arith::ExtSIOp>(loc, carrierType, scaled);
    if (op.getPreShiftLeft() != 0) {
      // Exact by construction: the carrier grew by exactly the shift amount.
      Value amount = createIntegerConstant(loc, carrierType, op.getPreShiftLeft(), rewriter);
      scaled = rewriter.create<arith::ShLIOp>(loc, scaled, amount);
    }

    Value divisor = createIntegerConstant(loc, carrierType, op.getDivisor(), rewriter);
    Value zero = createIntegerConstant(loc, carrierType, 0, rewriter);
    Value one = createIntegerConstant(loc, carrierType, 1, rewriter);
    // arith division truncates toward zero; the contract's Euclidean pair is
    // recovered from it on demand. For a positive divisor the truncated
    // remainder is negative exactly when the floor correction applies, so
    // that one sign test replaces the scaled < 0 && r != 0 form. Each mode
    // builds only the values it reads: toward_zero IS the truncated
    // quotient, toward_negative needs the corrected quotient alone, and only
    // the nearest modes need the non-negative remainder.
    Value truncated = rewriter.create<arith::DivSIOp>(loc, scaled, divisor);
    Value truncatedRemainder;
    Value needsCorrection;
    auto flooredQuotient = [&]() {
      truncatedRemainder = rewriter.create<arith::RemSIOp>(loc, scaled, divisor);
      needsCorrection =
          rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, truncatedRemainder, zero);
      Value stepped = rewriter.create<arith::SubIOp>(loc, truncated, one);
      return rewriter.create<arith::SelectOp>(loc, needsCorrection, stepped, truncated).getResult();
    };
    auto euclideanRemainder = [&]() {
      Value lifted = rewriter.create<arith::AddIOp>(loc, truncatedRemainder, divisor);
      return rewriter.create<arith::SelectOp>(loc, needsCorrection, lifted, truncatedRemainder)
          .getResult();
    };

    Value rounded;
    switch (op.getRounding()) {
    case ondrix::ondsp::RoundingMode::TowardNegative:
      rounded = flooredQuotient();
      break;
    case ondrix::ondsp::RoundingMode::TowardZero:
      // q + [scaled < 0 and r != 0] is the truncated quotient by construction.
      rounded = truncated;
      break;
    case ondrix::ondsp::RoundingMode::NearestTiesPositive: {
      // r >= divisor - r, stated without ever forming 2r.
      Value quotient = flooredQuotient();
      Value remainder = euclideanRemainder();
      Value complement = rewriter.create<arith::SubIOp>(loc, divisor, remainder);
      Value atLeastHalf =
          rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sge, remainder, complement);
      Value increment = rewriter.create<arith::SelectOp>(loc, atLeastHalf, one, zero);
      rounded = rewriter.create<arith::AddIOp>(loc, quotient, increment);
      break;
    }
    case ondrix::ondsp::RoundingMode::NearestEven: {
      Value quotient = flooredQuotient();
      Value remainder = euclideanRemainder();
      Value complement = rewriter.create<arith::SubIOp>(loc, divisor, remainder);
      Value aboveHalf =
          rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sgt, remainder, complement);
      Value exactlyHalf =
          rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, remainder, complement);
      Value quotientLowBit = rewriter.create<arith::AndIOp>(loc, quotient, one);
      Value quotientIsOdd =
          rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ne, quotientLowBit, zero);
      Value halfAndOdd = rewriter.create<arith::AndIOp>(loc, exactlyHalf, quotientIsOdd);
      Value condition = rewriter.create<arith::OrIOp>(loc, aboveHalf, halfAndOdd);
      Value increment = rewriter.create<arith::SelectOp>(loc, condition, one, zero);
      rounded = rewriter.create<arith::AddIOp>(loc, quotient, increment);
      break;
    }
    }

    unsigned destinationWidth = getIntegerElementType(op.getResult().getType()).getWidth();
    rewriter.replaceOp(
        op,
        narrowSignedValue(loc, rounded, getIntegerTypeLike(carrierType, destinationWidth, rewriter),
                          op.getOverflow(), rewriter));
    return success();
  }
};

class SatCastOpLowering final : public OpConversionPattern<ondrix::ondsp::SatCastOp> {
public:
  using OpConversionPattern<ondrix::ondsp::SatCastOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(ondrix::ondsp::SatCastOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto numeric = dyn_cast<ondrix::ondsp::FixedAttr>(op.getNumeric());
    if (!numeric || numeric.getSignedness() != ondrix::ondsp::Signedness::Signed)
      return op.emitOpError("fixed scalar lowering supports signed fixed sat_cast policies");
    IntegerType inputElement = getSignlessIntegerElementOrNull(adaptor.getInput().getType());
    if (!inputElement)
      return op.emitOpError("fixed scalar lowering supports scalar or fixed-width vector "
                            "signless integer sat_cast values");

    Location loc = op.getLoc();
    auto storage = cast<IntegerType>(numeric.getStorage());
    Type destinationType =
        getIntegerTypeLike(adaptor.getInput().getType(), storage.getWidth(), rewriter);
    Value result;
    if (storage.getWidth() > inputElement.getWidth())
      result = rewriter.create<arith::ExtSIOp>(loc, destinationType, adaptor.getInput());
    else
      result = narrowSignedValue(loc, adaptor.getInput(), destinationType,
                                 ondrix::ondsp::OverflowMode::Saturate, rewriter);
    rewriter.replaceOp(op, result);
    return success();
  }
};

class SqrtFixedOpLowering final : public OpConversionPattern<ondrix::ondsp::SqrtFixedOp> {
public:
  SqrtFixedOpLowering(TypeConverter &typeConverter, MLIRContext *context, bool sqrtEstimate)
      : OpConversionPattern(typeConverter, context), sqrtEstimate(sqrtEstimate) {}

  LogicalResult matchAndRewrite(ondrix::ondsp::SqrtFixedOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    // The registered lowering implements only the proven profile: a scalar
    // i64 sum of two Q1.15 squares (at most 2^31) to an i16 root. Other
    // widths and vector forms fail closed until a real consumer defines
    // their exact semantics and range proofs.
    auto inputType = dyn_cast<IntegerType>(adaptor.getInput().getType());
    if (!inputType || !inputType.isSignlessInteger(64))
      return op.emitOpError("fixed scalar lowering supports scalar i64 sqrt_fixed input");
    auto resultType = dyn_cast<IntegerType>(op.getResult().getType());
    if (!resultType || !resultType.isSignlessInteger(16))
      return op.emitOpError("fixed scalar lowering supports scalar i16 sqrt_fixed results");

    // The value domain is non-negative, but that producer obligation is not
    // decidable for arbitrary SSA input. The lowering therefore rejects an
    // input that is provably negative, and clamps the runtime value to zero
    // from below so that every out-of-domain execution deterministically
    // yields 0 — without the clamp the estimate branch would take the square
    // root of a negative value (NaN, then poison at the integer conversion).
    APInt constantInput;
    if (matchPattern(adaptor.getInput(), m_ConstantInt(&constantInput)) &&
        constantInput.isNegative())
      return op.emitOpError("constant input is negative and outside the sqrt_fixed value domain");

    Location loc = op.getLoc();
    Value zero = rewriter.create<arith::ConstantIntOp>(loc, 0, 64);
    Value input = rewriter.create<arith::MaxSIOp>(loc, adaptor.getInput(), zero);
    Value root;
    if (sqrtEstimate) {
      // The correction argument is the sqrt-estimate paragraph of this pass's
      // description; it covers inputs below 2^32, where the binary64
      // conversion is exact. At or above 2^32 the ceiling below pins the
      // candidate at 2^16, whose square never exceeds the input, so the
      // downward corrections cannot fire and the export saturates to 32767
      // under either definition — an inexact conversion never reaches output.
      Value one = rewriter.create<arith::ConstantIntOp>(loc, 1, 64);
      Value asFloat = rewriter.create<arith::SIToFPOp>(loc, rewriter.getF64Type(), input);
      Value estimate = rewriter.create<math::SqrtOp>(loc, asFloat);
      root = rewriter.create<arith::FPToSIOp>(loc, rewriter.getIntegerType(64), estimate);
      // Clamp the estimate before squaring: every root of at least 2^16
      // saturates to 32767 anyway, and the clamp keeps the correction
      // squares far from i64 overflow for the whole i64 input domain.
      Value ceiling = rewriter.create<arith::ConstantIntOp>(loc, 65536, 64);
      Value overCeiling =
          rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sgt, root, ceiling);
      root = rewriter.create<arith::SelectOp>(loc, overCeiling, ceiling, root);
      for (int step = 0; step < 2; ++step) {
        Value square = rewriter.create<arith::MulIOp>(loc, root, root);
        Value tooHigh =
            rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sgt, square, input);
        Value lowered = rewriter.create<arith::SubIOp>(loc, root, one);
        root = rewriter.create<arith::SelectOp>(loc, tooHigh, lowered, root);
      }
      for (int step = 0; step < 2; ++step) {
        Value next = rewriter.create<arith::AddIOp>(loc, root, one);
        Value square = rewriter.create<arith::MulIOp>(loc, next, next);
        Value fits = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sle, square, input);
        root = rewriter.create<arith::SelectOp>(loc, fits, next, root);
      }
    } else {
      root = rewriter.create<arith::ConstantIntOp>(loc, 0, 64);
      // Exact bit-by-bit integer square root: the root of an input bounded by
      // 2^31 fits in 16 bits, so 16 unrolled candidate bits suffice.
      for (int bit = 15; bit >= 0; --bit) {
        Value candidateBit = rewriter.create<arith::ConstantIntOp>(loc, int64_t(1) << bit, 64);
        Value candidate = rewriter.create<arith::AddIOp>(loc, root, candidateBit);
        Value square = rewriter.create<arith::MulIOp>(loc, candidate, candidate);
        Value fits = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sle, square, input);
        root = rewriter.create<arith::SelectOp>(loc, fits, candidate, root);
      }
    }
    if (op.getRounding() == ondrix::ondsp::RoundingMode::NearestEven) {
      // Round up when input - root^2 > root, i.e. sqrt(input) > root + 1/2.
      // (root + 1/2)^2 is never an integer, so there is no reachable tie.
      Value square = rewriter.create<arith::MulIOp>(loc, root, root);
      Value remainder = rewriter.create<arith::SubIOp>(loc, input, square);
      Value roundUp =
          rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sgt, remainder, root);
      Value one = rewriter.create<arith::ConstantIntOp>(loc, 1, 64);
      Value incremented = rewriter.create<arith::AddIOp>(loc, root, one);
      root = rewriter.create<arith::SelectOp>(loc, roundUp, incremented, root);
    }
    Value maximum = rewriter.create<arith::ConstantIntOp>(loc, 32767, 64);
    Value overflows = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sgt, root, maximum);
    Value clamped = rewriter.create<arith::SelectOp>(loc, overflows, maximum, root);
    rewriter.replaceOp(
        op, rewriter.create<arith::TruncIOp>(loc, rewriter.getI16Type(), clamped).getResult());
    return success();
  }

private:
  bool sqrtEstimate;
};

class ConvertOndspFixedToScalarPass final
    : public ondrix::impl::ConvertOndspFixedToScalarBase<ConvertOndspFixedToScalarPass> {
public:
  using ondrix::impl::ConvertOndspFixedToScalarBase<
      ConvertOndspFixedToScalarPass>::ConvertOndspFixedToScalarBase;

  void runOnOperation() override {
    if (failed(verifySourceArtifactUsage(getOperation()))) {
      signalPassFailure();
      return;
    }
    if (failed(ondrix::conversion::verifySCFWhileTypeConversionSafety(
            getOperation(), [](Type type) { return isa<ondrix::ondsp::AccType>(type); }))) {
      signalPassFailure();
      return;
    }

    OndspFixedToScalarTypeConverter typeConverter;
    RewritePatternSet patterns(&getContext());
    patterns.add<AccAddTermOpLowering, AccExportOpLowering, AccImportOpLowering, AccZeroOpLowering,
                 AddShiftOpLowering, MacOpLowering, MacSubOpLowering, ReduceMacOpLowering,
                 RoundDivOpLowering, RoundShiftOpLowering, SatCastOpLowering, SubShiftOpLowering>(
        typeConverter, &getContext());
    patterns.add<SqrtFixedOpLowering>(typeConverter, &getContext(), sqrtEstimate);
    patterns.add<CxButterflyOpLowering>(typeConverter, &getContext(), specializeCanonicalTwiddles);
    ondrix::conversion::populateValueTypeConversionPatterns(typeConverter, patterns);
    populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(patterns, typeConverter);
    populateCallOpTypeConversionPattern(patterns, typeConverter);
    populateBranchOpInterfaceTypeConversionPattern(patterns, typeConverter);
    populateReturnOpTypeConversionPattern(patterns, typeConverter);

    ConversionTarget target(getContext());
    target.addIllegalDialect<ondrix::ondsp::OndspDialect>();
    scf::populateSCFStructuralTypeConversionsAndLegality(typeConverter, patterns, target);
    target.addDynamicallyLegalOp<UnrealizedConversionCastOp>([](UnrealizedConversionCastOp op) {
      return !containsOndspAccumulator(op.getOperandTypes()) &&
             !containsOndspAccumulator(op.getResultTypes());
    });
    target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
      return typeConverter.isSignatureLegal(op.getFunctionType()) &&
             typeConverter.isLegal(&op.getBody()) &&
             hasLegalConvertedTypes(op.getOperation(), typeConverter);
    });
    target.addDynamicallyLegalOp<func::CallOp>(
        [&](func::CallOp op) { return hasLegalConvertedTypes(op, typeConverter); });
    target.addDynamicallyLegalOp<func::ReturnOp>(
        [&](func::ReturnOp op) { return hasLegalConvertedTypes(op, typeConverter); });
    target.markUnknownOpDynamicallyLegal([&](Operation *op) {
      bool hasLegalControlFlow =
          isNotBranchOpInterfaceOrReturnLikeOp(op) ||
          isLegalForBranchOpInterfaceTypeConversionPattern(op, typeConverter) ||
          isLegalForReturnOpTypeConversionPattern(op, typeConverter);
      return hasLegalControlFlow && hasLegalConvertedTypes(op, typeConverter);
    });

    if (failed(applyFullConversion(getOperation(), target, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> ondrix::createConvertOndspFixedToScalarPass() {
  return std::make_unique<ConvertOndspFixedToScalarPass>();
}
