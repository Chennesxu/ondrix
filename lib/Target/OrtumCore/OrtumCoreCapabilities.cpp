#include "ondrix/Target/OrtumCore/OrtumCoreCapabilities.h"

namespace ondrix::ortumcore {

AccumulatorDomain getSignedI40Frac30SaturatingAccumulatorDomain() {
  return {40, 30, ondsp::Signedness::Signed, ondsp::OverflowMode::Saturate};
}

ProductDomain getSignedQ15FullProductDomain() {
  return {16, 15, ondsp::Signedness::Signed,
          ondsp::ProductSemantics{32, 30, ondsp::ProductSelection::Full}};
}

ProductDomain getSignedQ31RawHighProductDomain() {
  return {32, 31, ondsp::Signedness::Signed,
          ondsp::ProductSemantics{32, 30, ondsp::ProductSelection::HighRaw}};
}

ExportDomain getShiftedSaturatingI32ExportDomain() {
  return {32, 15, ondsp::RoundingMode::TowardNegative, ondsp::OverflowMode::Saturate};
}

ScaledBinaryDomain getShiftedSaturatingI32ScaledBinaryDomain() {
  return {32, 3, ondsp::RoundingMode::TowardNegative, ondsp::OverflowMode::Saturate};
}

} // namespace ondrix::ortumcore
