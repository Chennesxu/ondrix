#include "ondrix/Target/OrtumCore/OrtumCoreTargetProfile.h"

#include "llvm/Support/raw_ostream.h"

using ondrix::ondsp::OverflowMode;
using ondrix::ondsp::ProductBitSelection;
using ondrix::ondsp::Signedness;
using ondrix::ortumcore::AccumulatorDomain;
using ondrix::ortumcore::OrtumCoreTargetProfile;
using ondrix::ortumcore::ProductDomain;

int main() {
  OrtumCoreTargetProfile profile;
  AccumulatorDomain targetAccumulator{40, 30, Signedness::Signed, OverflowMode::Saturate};
  ProductDomain q15Full{16, 15, Signedness::Signed, ProductBitSelection::Full};

  bool passed = profile.supportsAccumulator(targetAccumulator) &&
                profile.supportsMac(q15Full, targetAccumulator);
  passed &= !profile.supportsAccumulator(
      AccumulatorDomain{40, 30, Signedness::Signed, OverflowMode::Wrap});
  passed &= !profile.supportsAccumulator(
      AccumulatorDomain{40, 31, Signedness::Signed, OverflowMode::Saturate});
  passed &= !profile.supportsMac(
      ProductDomain{16, 15, Signedness::Signed, ProductBitSelection::HighRaw}, targetAccumulator);
  passed &= !profile.supportsMac(
      ProductDomain{32, 31, Signedness::Signed, ProductBitSelection::Full}, targetAccumulator);
  passed &= !profile.supportsMac(
      ProductDomain{32, 31, Signedness::Signed, ProductBitSelection::HighRaw}, targetAccumulator);

  if (!passed) {
    llvm::errs() << "ortumcore target profile: FAIL\n";
    return 1;
  }
  llvm::outs() << "ortumcore target profile: PASS\n";
  return 0;
}
