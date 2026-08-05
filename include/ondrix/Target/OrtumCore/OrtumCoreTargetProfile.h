#ifndef ONDRIX_TARGET_ORTUMCORE_ORTUMCORETARGETPROFILE_H
#define ONDRIX_TARGET_ORTUMCORE_ORTUMCORETARGETPROFILE_H

#include "ondrix/Target/OrtumCore/OrtumCoreCapabilities.h"

namespace ondrix::ortumcore {

/// Public semantic capabilities accepted by the OrtumCore adapter.
class OrtumCoreTargetProfile {
public:
  bool supportsAccumulator(const AccumulatorDomain &accumulator) const;
  bool supportsMac(const ProductDomain &product, const AccumulatorDomain &accumulator) const;
  /// True when an export of `accumulator` with this rounding/overflow policy
  /// and fractional shift is realizable through the readout path.
  bool supportsExport(const AccumulatorDomain &accumulator, ondsp::RoundingMode rounding,
                      ondsp::OverflowMode overflow, int64_t shift) const;
};

} // namespace ondrix::ortumcore

#endif // ONDRIX_TARGET_ORTUMCORE_ORTUMCORETARGETPROFILE_H
