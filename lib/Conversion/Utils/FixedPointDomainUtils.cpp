#include "ondrix/Conversion/Utils/FixedPointDomainUtils.h"

using namespace mlir;

namespace ondrix::conversion {

static bool isSignedQ15FullDomain(ondrix::ondsp::AccType accumulator,
                                  ondrix::ondsp::FixedAttr numeric,
                                  ondrix::ondsp::ProductAttr product) {
  if (ondrix::ondsp::isSignedQ15(numeric))
    return ondrix::ondsp::isFullProduct(product) &&
           ondrix::ondsp::isSignedI40Frac30Accumulator(accumulator);
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
  return isSignedQ15FullDomain(accumulator, numeric, product) ||
         isSignedQ31FullDomain(accumulator, numeric, product) ||
         isSignedQ31RawHighDomain(accumulator, numeric, product);
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
