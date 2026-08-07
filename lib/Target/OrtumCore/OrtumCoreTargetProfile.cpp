#include "ondrix/Target/OrtumCore/OrtumCoreTargetProfile.h"

namespace ondrix::ortumcore {
namespace {

static bool matches(const AccumulatorDomain &lhs, const AccumulatorDomain &rhs) {
  return lhs.storageWidth == rhs.storageWidth && lhs.frac == rhs.frac &&
         lhs.signedness == rhs.signedness && lhs.updateOverflow == rhs.updateOverflow;
}

static bool matches(const ProductDomain &lhs, const ProductDomain &rhs) {
  return lhs.operandWidth == rhs.operandWidth && lhs.operandFrac == rhs.operandFrac &&
         lhs.signedness == rhs.signedness && lhs.product.rawWidth == rhs.product.rawWidth &&
         lhs.product.frac == rhs.product.frac && lhs.product.selection == rhs.product.selection;
}

} // namespace

bool OrtumCoreTargetProfile::supportsAccumulator(const AccumulatorDomain &accumulator) const {
  return matches(accumulator, getSignedI40Frac30SaturatingAccumulatorDomain());
}

bool OrtumCoreTargetProfile::supportsMac(const ProductDomain &product,
                                         const AccumulatorDomain &accumulator) const {
  return supportsAccumulator(accumulator) && matches(product, getSignedQ15FullProductDomain());
}

bool OrtumCoreTargetProfile::supportsExport(const AccumulatorDomain &accumulator,
                                            ondsp::RoundingMode rounding,
                                            ondsp::OverflowMode overflow, int64_t shift) const {
  ExportDomain domain = getShiftedSaturatingI32ExportDomain();
  if (!supportsAccumulator(accumulator) || overflow != domain.overflow)
    return false;
  if (rounding == domain.rounding)
    return shift >= 0 && shift <= int64_t(domain.maxShift);
  // Ties-positive is realized as the floor readout at shift-1 followed by
  // one increment-and-halve; exact only when the narrower readout provably
  // cannot clip (shift >= width - 32 + 1), vacuous at shift 0.
  if (rounding == ondsp::RoundingMode::NearestTiesPositive)
    return shift == 0 ||
           (shift >= int64_t(accumulator.storageWidth) - int64_t(domain.storageWidth) + 1 &&
            shift <= int64_t(domain.maxShift));
  return false;
}

} // namespace ondrix::ortumcore
