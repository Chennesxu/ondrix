#include "ondrix/Target/OrtumCore/OrtumCoreTargetProfile.h"

namespace ondrix::ortumcore {

bool OrtumCoreTargetProfile::supportsAccumulator(const AccumulatorDomain &accumulator) const {
  return accumulator.storageWidth == 40 && accumulator.frac == 30 &&
         accumulator.signedness == ondsp::Signedness::Signed &&
         accumulator.updateOverflow == ondsp::OverflowMode::Saturate;
}

bool OrtumCoreTargetProfile::supportsMac(const ProductDomain &product,
                                         const AccumulatorDomain &accumulator) const {
  return supportsAccumulator(accumulator) && product.operandWidth == 16 &&
         product.operandFrac == 15 && product.signedness == ondsp::Signedness::Signed &&
         product.selection == ondsp::ProductBitSelection::Full;
}

} // namespace ondrix::ortumcore
