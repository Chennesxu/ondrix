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
  return supportsAccumulator(accumulator) && rounding == domain.rounding &&
         overflow == domain.overflow && shift >= 0 && shift <= int64_t(domain.maxShift);
}

} // namespace ondrix::ortumcore
