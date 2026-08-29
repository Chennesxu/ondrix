#include "ondrix/Target/OrtumCore/OrtumCoreTargetProfile.h"

#include "llvm/Support/raw_ostream.h"

using ondrix::ondsp::OverflowMode;
using ondrix::ondsp::ProductSelection;
using ondrix::ondsp::ProductSemantics;
using ondrix::ondsp::Signedness;
using ondrix::ortumcore::AccumulatorDomain;
using ondrix::ortumcore::OrtumCoreTargetProfile;
using ondrix::ortumcore::ProductDomain;

int main() {
  OrtumCoreTargetProfile profile;
  AccumulatorDomain targetAccumulator{40, 30, Signedness::Signed, OverflowMode::Saturate};
  ProductDomain q15Full{16, 15, Signedness::Signed,
                        ProductSemantics{32, 30, ProductSelection::Full}};

  bool passed = profile.supportsAccumulator(targetAccumulator) &&
                profile.supportsMac(q15Full, targetAccumulator);
  passed &= !profile.supportsAccumulator(
      AccumulatorDomain{40, 30, Signedness::Signed, OverflowMode::Wrap});
  passed &= !profile.supportsAccumulator(
      AccumulatorDomain{40, 31, Signedness::Signed, OverflowMode::Saturate});
  passed &= !profile.supportsMac(ProductDomain{16, 15, Signedness::Signed,
                                               ProductSemantics{16, 14, ProductSelection::HighRaw}},
                                 targetAccumulator);
  passed &= !profile.supportsMac(
      ProductDomain{32, 31, Signedness::Signed, ProductSemantics{64, 62, ProductSelection::Full}},
      targetAccumulator);
  passed &= !profile.supportsMac(ProductDomain{32, 31, Signedness::Signed,
                                               ProductSemantics{32, 30, ProductSelection::HighRaw}},
                                 targetAccumulator);

  // The composed export ladder: shifts past 15 ride the always-exact
  // shift-15 readout plus a base tail of at most 31; NTP inside (0, 9)
  // has no exact composition and stays refused.
  using ondrix::ondsp::RoundingMode;
  passed &= profile.supportsExport(targetAccumulator, RoundingMode::TowardNegative,
                                   OverflowMode::Saturate, 15);
  passed &= profile.supportsExport(targetAccumulator, RoundingMode::TowardNegative,
                                   OverflowMode::Saturate, 22);
  passed &= profile.supportsExport(targetAccumulator, RoundingMode::TowardNegative,
                                   OverflowMode::Saturate, 46);
  passed &= !profile.supportsExport(targetAccumulator, RoundingMode::TowardNegative,
                                    OverflowMode::Saturate, 47);
  passed &= profile.supportsExport(targetAccumulator, RoundingMode::NearestTiesPositive,
                                   OverflowMode::Saturate, 22);
  passed &= !profile.supportsExport(targetAccumulator, RoundingMode::NearestTiesPositive,
                                    OverflowMode::Saturate, 5);
  passed &= !profile.supportsExport(targetAccumulator, RoundingMode::NearestEven,
                                    OverflowMode::Saturate, 15);

  if (!passed) {
    llvm::errs() << "ortumcore target profile: FAIL\n";
    return 1;
  }
  llvm::outs() << "ortumcore target profile: PASS\n";
  return 0;
}
