#include "ondrix/Conversion/Utils/FixedPointDomainUtils.h"

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
  return ondrix::ondsp::isSignedQ31(numeric) && ondrix::ondsp::isFullProduct(product) &&
         ondrix::ondsp::isSignedI64Frac62Accumulator(accumulator);
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

} // namespace ondrix::conversion
