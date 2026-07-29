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
#include "ondrix/Support/FixedPointSemantics.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
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
  // The lowering body is already width and shift generic: it derives
  // `shift = acc.frac - dst.frac` and routes it through
  // `roundSignedRightShift` and `narrowSignedValue`. This gate used to be
  // narrower than the body and admitted only frac 30 for an i32 destination,
  // so any signed i32 destination whose fractional position the verifier
  // already admits (`dst.frac <= acc.frac`) now lowers as well. That widened
  // domain carries requantizations whose shift is not the Q-format difference,
  // such as the mean boundary of `ondrix.rms`, and is covered by an
  // object-level differential gate. Every other destination stays refused.
  return ondrix::ondsp::isSignedQ15(destination) || isSignedFixedStorage(destination, 32);
}

static bool isSupportedAccumulatorTerm(ondrix::ondsp::AccType accumulator,
                                       ondrix::ondsp::FixedAttr numeric) {
  auto storage = dyn_cast<IntegerType>(numeric.getStorage());
  return isSupportedAccumulator(accumulator) && storage && storage.isSignless() &&
         numeric.getSignedness() == ondrix::ondsp::Signedness::Signed &&
         numeric.getFrac() == accumulator.getFrac();
}

class OndspFixedToScalarTypeConverter final : public TypeConverter {
public:
  OndspFixedToScalarTypeConverter() {
    addConversion([](Type type) { return type; });
    addConversion([](ondrix::ondsp::AccType type,
                     SmallVectorImpl<Type> &results) -> std::optional<LogicalResult> {
      if (!isSupportedAccumulator(type))
        return failure();
      results.push_back(type.getStorage());
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

class AccZeroOpLowering final : public OpConversionPattern<ondrix::ondsp::AccZeroOp> {
public:
  using OpConversionPattern<ondrix::ondsp::AccZeroOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(ondrix::ondsp::AccZeroOp op, OpAdaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Type resultType = getTypeConverter()->convertType(op.getAcc().getType());
    auto integerType = dyn_cast_or_null<IntegerType>(resultType);
    if (!integerType)
      return op.emitOpError("fixed scalar lowering requires a supported accumulator type");
    rewriter.replaceOpWithNewOp<arith::ConstantOp>(op, integerType,
                                                   rewriter.getIntegerAttr(integerType, 0));
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
  auto accumulatorType = cast<IntegerType>(accumulator.getType());
  auto productType = cast<IntegerType>(product.getType());
  unsigned intermediateWidth = ondrix::fixedpoint::getAccumulatorUpdateIntermediateWidth(
      accumulatorType.getWidth(), productType.getWidth());
  IntegerType intermediateType = builder.getIntegerType(intermediateWidth);
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

  // Clamp in the exact update width before narrowing to the accumulator.
  llvm::APInt minimum =
      llvm::APInt::getSignedMinValue(accumulatorType.getWidth()).sext(intermediateWidth);
  llvm::APInt maximum =
      llvm::APInt::getSignedMaxValue(accumulatorType.getWidth()).sext(intermediateWidth);
  Value minimumValue = builder.create<arith::ConstantOp>(
      loc, intermediateType, builder.getIntegerAttr(intermediateType, minimum));
  Value maximumValue = builder.create<arith::ConstantOp>(
      loc, intermediateType, builder.getIntegerAttr(intermediateType, maximum));
  Value belowMinimum =
      builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, updated, minimumValue);
  Value aboveMaximum =
      builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sgt, updated, maximumValue);
  Value lowerClamped = builder.create<arith::SelectOp>(loc, belowMinimum, minimumValue, updated);
  Value clamped = builder.create<arith::SelectOp>(loc, aboveMaximum, maximumValue, lowerClamped);
  return builder.create<arith::TruncIOp>(loc, accumulatorType, clamped);
}

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
                                   ConversionPatternRewriter &rewriter) {
  IntegerType elementType = getIntegerElementType(type);
  IntegerAttr valueAttr = rewriter.getIntegerAttr(elementType, value);
  if (auto vector = dyn_cast<VectorType>(type))
    return rewriter.create<arith::ConstantOp>(loc, vector,
                                              SplatElementsAttr::get(vector, valueAttr));
  return rewriter.create<arith::ConstantOp>(loc, type, valueAttr);
}

static Value createIntegerConstant(Location loc, Type type, int64_t value,
                                   ConversionPatternRewriter &rewriter) {
  return createIntegerConstant(
      loc, type, llvm::APInt(getIntegerElementType(type).getWidth(), value, true), rewriter);
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

static std::pair<Value, Value> unpackPackedQ15(Location loc, Value packed,
                                               ConversionPatternRewriter &rewriter) {
  Type i16 = getIntegerTypeLike(packed.getType(), 16, rewriter);
  Type i32 = getIntegerTypeLike(packed.getType(), 32, rewriter);
  Value real = rewriter.create<arith::TruncIOp>(loc, i16, packed);
  Value shift = createIntegerConstant(loc, i32, 16, rewriter);
  Value high = rewriter.create<arith::ShRUIOp>(loc, packed, shift);
  Value imaginary = rewriter.create<arith::TruncIOp>(loc, i16, high);
  return {real, imaginary};
}

static Value packQ15Complex(Location loc, Value real, Value imaginary,
                            ConversionPatternRewriter &rewriter) {
  Type i32 = getIntegerTypeLike(real.getType(), 32, rewriter);
  Value realBits = rewriter.create<arith::ExtUIOp>(loc, i32, real);
  Value imaginaryBits = rewriter.create<arith::ExtUIOp>(loc, i32, imaginary);
  Value shift = createIntegerConstant(loc, i32, 16, rewriter);
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
  IntegerType fullProductType =
      builder.getIntegerType(cast<IntegerType>(numeric.getStorage()).getWidth() * 2);
  Value lhsExtended = builder.create<arith::ExtSIOp>(loc, fullProductType, lhs);
  Value rhsExtended = builder.create<arith::ExtSIOp>(loc, fullProductType, rhs);
  Value fullProduct = builder.create<arith::MulIOp>(loc, lhsExtended, rhsExtended);
  if (semantics.selection == ondrix::ondsp::ProductSelection::Full)
    return fullProduct;

  Value shift = builder.create<arith::ConstantIntOp>(
      loc, cast<IntegerType>(numeric.getStorage()).getWidth(), fullProductType.getWidth());
  Value shifted = builder.create<arith::ShRSIOp>(loc, fullProduct, shift);
  IntegerType rawHighType = builder.getIntegerType(semantics.rawWidth);
  return builder.create<arith::TruncIOp>(loc, rawHighType, shifted);
}

template <typename OpTy, ondrix::fixedpoint::AccumulatorUpdateOperation operation>
class MacLikeOpLowering final : public OpConversionPattern<OpTy> {
public:
  using OpConversionPattern<OpTy>::OpConversionPattern;

  LogicalResult matchAndRewrite(OpTy op, typename OpTy::Adaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto accumulator = cast<ondrix::ondsp::AccType>(op.getAcc().getType());
    FailureOr<ondrix::conversion::SupportedFixedMacDomain> domain =
        ondrix::conversion::getSupportedFixedScalarMacDomain(op, accumulator, op.getNumeric(),
                                                             op.getProduct());
    if (failed(domain))
      return op.emitOpError(
          "fixed scalar lowering supports Q15/full with a signed frac30 accumulator of at least "
          "32 bits, Q31/full with i64/frac62, or Q31/high_raw with i40/frac30 accumulation");

    Value product = lowerSignedProduct(op.getLoc(), adaptor.getLhs(), adaptor.getRhs(),
                                       op.getNumeric(), domain->product, rewriter);
    Value updated = lowerAccumulatorUpdate(op.getLoc(), adaptor.getAcc(), product,
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
    auto [aReal, aImaginary] = unpackPackedQ15(loc, adaptor.getA(), rewriter);
    auto [bReal, bImaginary] = unpackPackedQ15(loc, adaptor.getB(), rewriter);
    auto [wReal, wImaginary] = unpackPackedQ15(loc, adaptor.getTwiddle(), rewriter);

    Value twiddledReal;
    Value twiddledImaginary;
    bool specialized = false;
    if (specializeCanonicalTwiddles) {
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
      Type productType = getIntegerTypeLike(bReal.getType(), 33, rewriter);
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

    Type sumType = getIntegerTypeLike(aReal.getType(), 17, rewriter);
    auto extendSumOperand = [&](Value value) {
      return rewriter.create<arith::ExtSIOp>(loc, sumType, value);
    };
    Value ar = extendSumOperand(aReal);
    Value ai = extendSumOperand(aImaginary);
    Value tr = extendSumOperand(twiddledReal);
    Value ti = extendSumOperand(twiddledImaginary);
    Value out0Real = rewriter.create<arith::AddIOp>(loc, ar, tr);
    Value out0Imaginary = rewriter.create<arith::AddIOp>(loc, ai, ti);
    Value out1Real = rewriter.create<arith::SubIOp>(loc, ar, tr);
    Value out1Imaginary = rewriter.create<arith::SubIOp>(loc, ai, ti);

    out0Real = requantizeSignedValue(loc, out0Real, op.getOutputScale(), rewriter);
    out0Imaginary = requantizeSignedValue(loc, out0Imaginary, op.getOutputScale(), rewriter);
    out1Real = requantizeSignedValue(loc, out1Real, op.getOutputScale(), rewriter);
    out1Imaginary = requantizeSignedValue(loc, out1Imaginary, op.getOutputScale(), rewriter);

    rewriter.replaceOp(op, ValueRange{packQ15Complex(loc, out0Real, out0Imaginary, rewriter),
                                      packQ15Complex(loc, out1Real, out1Imaginary, rewriter)});
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
          "signed Q15 or signed i32 destination, and i64/frac62 to Q31 export");

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
    auto destinationType = cast<IntegerType>(op.getDst().getStorage());
    Value result =
        narrowSignedValue(op.getLoc(), rounded, destinationType, op.getOverflow(), rewriter);
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
      // Bit-exact over the whole clamped i64 domain, in two cases. Below
      // 2^32 the conversion to binary64 is exact and IEEE 754 requires a
      // correctly rounded square root, so the truncated estimate is within
      // one of the exact integer floor; two branchless correction steps in
      // each direction give double that margin. At or above 2^32 the exact
      // root is at least 2^16, the correctly rounded estimate is too, and
      // the ceiling below pins the candidate at exactly 2^16 (its square,
      // 2^32, never exceeds the input, so the downward corrections cannot
      // fire); every final candidate then stays above the i16 maximum and
      // the observable result is the saturated 32767 in both definitions.
      // The inexact binary64 conversion of large inputs never reaches the
      // output, so no exact-representability precondition is required. The
      // opt-in pass option assumes the target provides an IEEE 754
      // correctly rounded binary64 square root.
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
                 MacOpLowering, MacSubOpLowering, ReduceMacOpLowering, RoundShiftOpLowering,
                 SatCastOpLowering>(typeConverter, &getContext());
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
