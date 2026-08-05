#ifndef ONDRIX_TARGET_ORTUMCORE_ORTUMCORECAPABILITIES_H
#define ONDRIX_TARGET_ORTUMCORE_ORTUMCORECAPABILITIES_H

#include "ondrix/Dialect/ondsp/IR/OndspSemantics.h"

namespace ondrix::ortumcore {

/// Target-independent accumulator properties used for capability matching.
struct AccumulatorDomain {
  unsigned storageWidth;
  unsigned frac;
  ondsp::Signedness signedness;
  ondsp::OverflowMode updateOverflow;
};

/// Target-independent operand and raw product properties used for matching.
struct ProductDomain {
  unsigned operandWidth;
  unsigned operandFrac;
  ondsp::Signedness signedness;
  ondsp::ProductSemantics product;
};

/// Returns the numeric domain represented by parameterless `!ortumcore.acc`.
AccumulatorDomain getSignedI40Frac30SaturatingAccumulatorDomain();

/// Returns the product domain represented by OrtumCore MAC add/sub.
ProductDomain getSignedQ15FullProductDomain();

/// Target-independent properties of the accumulator readout path used for
/// export capability matching.
struct ExportDomain {
  unsigned storageWidth;
  unsigned maxShift;
  ondsp::RoundingMode rounding;
  ondsp::OverflowMode overflow;
};

/// Returns the export domain represented by OrtumCore acc_out: arithmetic
/// right shift in [0, maxShift], then saturation to the signed 32-bit range.
ExportDomain getShiftedSaturatingI32ExportDomain();

} // namespace ondrix::ortumcore

#endif // ONDRIX_TARGET_ORTUMCORE_ORTUMCORECAPABILITIES_H
