#ifndef ONDRIX_TARGET_ORTUMCORE_ORTUMCORETARGETPROFILE_H
#define ONDRIX_TARGET_ORTUMCORE_ORTUMCORETARGETPROFILE_H

#include "ondrix/Dialect/ondsp/IR/OndspEnums.h"
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
  ondsp::ProductBitSelection selection;
};

/// Public semantic capabilities accepted by the OrtumCore adapter.
class OrtumCoreTargetProfile {
public:
  bool supportsAccumulator(const AccumulatorDomain &accumulator) const;
  bool supportsMac(const ProductDomain &product, const AccumulatorDomain &accumulator) const;
};

} // namespace ondrix::ortumcore

#endif // ONDRIX_TARGET_ORTUMCORE_ORTUMCORETARGETPROFILE_H
