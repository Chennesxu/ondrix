#include "ondrix/Conversion/Utils/FixedPointDomainUtils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/TypeUtilities.h"

#include <algorithm>

using namespace mlir;

namespace ondrix::conversion {

static bool isSignedQ15FullDomain(ondrix::ondsp::AccType accumulator,
                                  ondrix::ondsp::FixedAttr numeric,
                                  ondrix::ondsp::ProductAttr product) {
  if (ondrix::ondsp::isSignedQ15(numeric))
    return ondrix::ondsp::isFullProduct(product) &&
           accumulator.getSignedness() == ondrix::ondsp::Signedness::Signed &&
           accumulator.getFrac() == 30 &&
           accumulator.getStorage().cast<IntegerType>().getWidth() >= 32;
  return false;
}

static bool isSignedQ31FullDomain(ondrix::ondsp::AccType accumulator,
                                  ondrix::ondsp::FixedAttr numeric,
                                  ondrix::ondsp::ProductAttr product) {
  // The accumulator sits at the term's fractional position: 62 for the exact
  // product, 62 - s for a product requantized by s.
  auto storage = dyn_cast<IntegerType>(accumulator.getStorage());
  return ondrix::ondsp::isSignedQ31(numeric) && ondrix::ondsp::isFullProduct(product) &&
         accumulator.getSignedness() == ondrix::ondsp::Signedness::Signed && storage &&
         storage.getWidth() == 64 && product.getShift() <= 62 &&
         accumulator.getFrac() == 62 - product.getShift();
}

static bool isSignedQ31RawHighDomain(ondrix::ondsp::AccType accumulator,
                                     ondrix::ondsp::FixedAttr numeric,
                                     ondrix::ondsp::ProductAttr product) {
  return ondrix::ondsp::isSignedQ31(numeric) && ondrix::ondsp::isRawHighProduct(product) &&
         ondrix::ondsp::isSignedI40Frac30Accumulator(accumulator);
}

bool isSupportedFixedScalarMacDomain(ondrix::ondsp::AccType accumulator,
                                     ondrix::ondsp::FixedAttr numeric,
                                     ondrix::ondsp::ProductAttr product) {
  // This predicate describes the PER-LANE arithmetic domain and is therefore
  // deliberately lane count independent: a multi-lane accumulator is W copies
  // of exactly this domain. Which operations may carry more than one lane is
  // an operation question the ondsp verifiers answer, not a domain question.
  return isSignedQ15FullDomain(accumulator, numeric, product) ||
         isSignedQ31FullDomain(accumulator, numeric, product) ||
         isSignedQ31RawHighDomain(accumulator, numeric, product);
}

bool isSupportedFixedVectorMacDomain(ondrix::ondsp::AccType accumulator,
                                     ondrix::ondsp::FixedAttr numeric,
                                     ondrix::ondsp::ProductAttr product) {
  // Vector lowering currently uses signed extension and signed high-half
  // selection. Keep signedness in the capability gate, not in rewrite details.
  if (numeric.getSignedness() != ondrix::ondsp::Signedness::Signed)
    return false;
  // The fixed-length Vector consumers spend the vector lanes on the REDUCTION
  // axis: they pair lane i of the value vector with lane i of the coefficient
  // vector and fold them into one accumulator. A multi-lane accumulator spends
  // its lanes on independent outputs instead, so the two lane meanings would
  // collide. Refuse them here rather than in each pass, and keep the refusal
  // separate from scalar legality so neither can widen the other.
  if (!ondrix::ondsp::isSingleLaneAccumulator(accumulator))
    return false;
  if (accumulator.getStorage().cast<IntegerType>().getWidth() > 64)
    return false;
  return isSignedQ15FullDomain(accumulator, numeric, product) ||
         isSignedQ31FullDomain(accumulator, numeric, product) ||
         isSignedQ31RawHighDomain(accumulator, numeric, product);
}

bool isSupportedFixedHorizontalMacDomain(ondrix::ondsp::AccType accumulator,
                                         ondrix::ondsp::FixedAttr numeric,
                                         ondrix::ondsp::ProductAttr product) {
  if (!isSupportedFixedVectorMacDomain(accumulator, numeric, product))
    return false;
  if (ondrix::ondsp::isSignedQ15(numeric)) {
    if (ondrix::ondsp::isSignedI40Frac30Accumulator(accumulator))
      return true;
    // Exact-modulo reassociation legality is width independent up to the i64
    // horizontal carrier: the lane terms are widened to i64 and summed by one
    // `vector.reduction<add>`, and modular i64 addition preserves every low
    // bit of any accumulator at most 64 bits wide. The i40 restriction above
    // exists for the saturating prefix-proof path, which gates independently
    // on Saturate update overflow in `planConstantSaturatingReduction` and
    // therefore cannot be widened here. The `> 40` cut below is deliberately
    // conservative, not a legality limit: it leaves the established i40
    // branch above (which admits both overflow modes) untouched and admits
    // only the wider wrapping accumulators that current producers need.
    // Narrower wrapping accumulators such as i34 stay refused and pinned by
    // the existing portable-profile negative.
    return accumulator.getUpdateOverflow() == ondrix::ondsp::OverflowMode::Wrap &&
           accumulator.getStorage().cast<IntegerType>().getWidth() > 40;
  }
  return true;
}

static FailureOr<SupportedFixedMacDomain>
materializeFixedMacDomain(Operation *op, ondrix::ondsp::FixedAttr numeric,
                          ondrix::ondsp::ProductAttr product) {

  FailureOr<ondrix::ondsp::ProductSemantics> semantics =
      ondrix::ondsp::inferProductSemantics(op, numeric, product);
  if (failed(semantics))
    return failure();

  auto operandStorage = cast<IntegerType>(numeric.getStorage());
  IntegerType fullProductStorage =
      IntegerType::get(numeric.getContext(), operandStorage.getWidth() * 2);
  IntegerType termStorage = IntegerType::get(numeric.getContext(), semantics->rawWidth);
  return SupportedFixedMacDomain{numeric.getSignedness(), operandStorage, fullProductStorage,
                                 termStorage, *semantics};
}

FailureOr<SupportedFixedMacDomain>
getSupportedFixedScalarMacDomain(Operation *op, ondrix::ondsp::AccType accumulator,
                                 ondrix::ondsp::FixedAttr numeric,
                                 ondrix::ondsp::ProductAttr product) {
  if (!isSupportedFixedScalarMacDomain(accumulator, numeric, product))
    return failure();
  return materializeFixedMacDomain(op, numeric, product);
}

FailureOr<SupportedFixedMacDomain>
getSupportedFixedVectorMacDomain(Operation *op, ondrix::ondsp::AccType accumulator,
                                 ondrix::ondsp::FixedAttr numeric,
                                 ondrix::ondsp::ProductAttr product) {
  if (!isSupportedFixedVectorMacDomain(accumulator, numeric, product))
    return failure();
  return materializeFixedMacDomain(op, numeric, product);
}

/// Whether `2^(valueBits-1) + 2^(shift-1)` is below the signed rail of `width`
/// bits, so the add-half of a ties-positive rounding cannot overflow.
static bool halfAddCannotOverflow(unsigned valueBits, unsigned shift, unsigned width) {
  if (valueBits == 0 || shift == 0 || valueBits >= width || shift >= width)
    return false;
  unsigned top = std::max(valueBits, shift);
  return valueBits == shift ? top + 1 < width : top < width;
}

Value createRoundedSignedRightShift(Location loc, Value input, unsigned shift,
                                    ondrix::ondsp::RoundingMode roundingMode, OpBuilder &builder,
                                    unsigned valueBits) {
  Type type = input.getType();
  if (shift == 0)
    return input;
  auto element = cast<IntegerType>(getElementTypeOrSelf(type));
  auto constant = [&](int64_t value) -> Value {
    Attribute attr = builder.getIntegerAttr(element, value);
    if (auto vector = dyn_cast<VectorType>(type))
      attr = SplatElementsAttr::get(vector, attr);
    return builder.create<arith::ConstantOp>(loc, type, cast<TypedAttr>(attr));
  };
  // Scalar only: on a vector the add-half lets LLVM narrow the surrounding
  // lane arithmetic, which x86 then legalizes worse than the remainder form.
  if (roundingMode == ondrix::ondsp::RoundingMode::NearestTiesPositive && !isa<VectorType>(type) &&
      halfAddCannotOverflow(valueBits, shift, element.getWidth())) {
    Value biased = builder.create<arith::AddIOp>(loc, input, constant(int64_t{1} << (shift - 1)));
    return builder.create<arith::ShRSIOp>(loc, biased, constant(shift));
  }
  Value quotient = builder.create<arith::ShRSIOp>(loc, input, constant(shift));
  if (roundingMode == ondrix::ondsp::RoundingMode::TowardNegative)
    return quotient;
  // The remainder is taken from the low `shift` bits, so no add-half in the
  // input width can overflow near its maximum; the increment is total.
  Type remainderBitsType = IntegerType::get(builder.getContext(), shift);
  if (auto vector = dyn_cast<VectorType>(type))
    remainderBitsType = VectorType::get(vector.getShape(), remainderBitsType);
  Value remainderBits = builder.create<arith::TruncIOp>(loc, remainderBitsType, input);
  Value remainder = builder.create<arith::ExtUIOp>(loc, type, remainderBits);
  Value zero = constant(0);
  Value one = constant(1);
  Value increment;
  switch (roundingMode) {
  case ondrix::ondsp::RoundingMode::TowardNegative:
    llvm_unreachable("toward-negative rounding returned above");
  case ondrix::ondsp::RoundingMode::TowardZero: {
    Value isNegative = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, input, zero);
    Value hasRemainder =
        builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ne, remainder, zero);
    increment = builder.create<arith::AndIOp>(loc, isNegative, hasRemainder);
    break;
  }
  case ondrix::ondsp::RoundingMode::NearestTiesPositive: {
    Value half = constant(int64_t{1} << (shift - 1));
    increment = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::uge, remainder, half);
    break;
  }
  case ondrix::ondsp::RoundingMode::NearestEven: {
    Value half = constant(int64_t{1} << (shift - 1));
    Value aboveHalf =
        builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ugt, remainder, half);
    // A tie with an odd quotient is one pattern of the low shift+1 bits of the
    // input, tested in the input width: a narrower lane type would be
    // legalized through packs on wide vectors.
    Value tieBits =
        shift + 1 < element.getWidth()
            ? builder
                  .create<arith::AndIOp>(
                      loc, input, constant(static_cast<int64_t>((uint64_t{1} << (shift + 1)) - 1)))
                  .getResult()
            : input;
    Value halfAndOdd = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, tieBits,
                                                     constant(int64_t{3} << (shift - 1)));
    increment = builder.create<arith::OrIOp>(loc, aboveHalf, halfAndOdd);
    break;
  }
  }
  Value incrementValue = builder.create<arith::SelectOp>(loc, increment, one, zero);
  return builder.create<arith::AddIOp>(loc, quotient, incrementValue);
}

} // namespace ondrix::conversion
