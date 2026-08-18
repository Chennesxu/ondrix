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

/// Returns the product domain represented by OrtumCore Q31 MAC add/sub: the
/// arithmetic high half of the exact signed 32-by-32 product. Its fractional
/// position is the accumulator's own, which is why the two MAC families share
/// one accumulator domain instead of each needing its own.
ProductDomain getSignedQ31RawHighProductDomain();

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

/// Target-independent properties of the scaled saturating add/sub path used
/// for capability matching. Its shift bound is independent of the readout
/// path's: the two are different instructions.
struct ScaledBinaryDomain {
  unsigned storageWidth;
  unsigned maxShift;
  ondsp::RoundingMode rounding;
  ondsp::OverflowMode overflow;
};

/// Returns the domain represented by OrtumCore sat_shift_add/sat_shift_sub:
/// the exact sum or difference shifts arithmetically right by an amount in
/// [0, maxShift], then saturates to the signed 32-bit range.
ScaledBinaryDomain getShiftedSaturatingI32ScaledBinaryDomain();

} // namespace ondrix::ortumcore

#endif // ONDRIX_TARGET_ORTUMCORE_ORTUMCORECAPABILITIES_H
