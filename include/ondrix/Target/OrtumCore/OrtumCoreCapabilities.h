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

} // namespace ondrix::ortumcore

#endif // ONDRIX_TARGET_ORTUMCORE_ORTUMCORECAPABILITIES_H
