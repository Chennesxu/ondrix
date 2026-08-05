#include "ondrix/Target/OrtumCore/OrtumCoreCapabilities.h"

namespace ondrix::ortumcore {

AccumulatorDomain getSignedI40Frac30SaturatingAccumulatorDomain() {
  return {40, 30, ondsp::Signedness::Signed, ondsp::OverflowMode::Saturate};
}

ProductDomain getSignedQ15FullProductDomain() {
  return {16, 15, ondsp::Signedness::Signed,
          ondsp::ProductSemantics{32, 30, ondsp::ProductSelection::Full}};
}

ExportDomain getShiftedSaturatingI32ExportDomain() {
  return {32, 15, ondsp::RoundingMode::TowardNegative, ondsp::OverflowMode::Saturate};
}

} // namespace ondrix::ortumcore
