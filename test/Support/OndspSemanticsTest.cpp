#include "ondrix/Dialect/ondsp/IR/OndspSemantics.h"

#include "llvm/Support/raw_ostream.h"

using ondrix::ondsp::classifyReductionReassociation;
using ondrix::ondsp::OverflowMode;
using ondrix::ondsp::ReductionRangeProof;
using ondrix::ondsp::ReductionReassociationSafety;

namespace {

bool testReductionReassociationSafety() {
  bool passed = true;
  passed &= classifyReductionReassociation(OverflowMode::Wrap) ==
            ReductionReassociationSafety::ExactModulo;
  passed &= classifyReductionReassociation(OverflowMode::Saturate) ==
            ReductionReassociationSafety::MustPreserveOrder;
  passed &= classifyReductionReassociation(
                OverflowMode::Saturate, ReductionRangeProof::AllOriginalAndReassociatedSumsFit) ==
            ReductionReassociationSafety::ProvenNoOverflow;
  passed &= classifyReductionReassociation(
                OverflowMode::Wrap, ReductionRangeProof::AllOriginalAndReassociatedSumsFit) ==
            ReductionReassociationSafety::ProvenNoOverflow;
  return passed;
}

} // namespace

int main() {
  if (!testReductionReassociationSafety()) {
    llvm::errs() << "ondsp reassociation semantics: FAIL\n";
    return 1;
  }
  llvm::outs() << "ondsp reassociation semantics: PASS\n";
  return 0;
}
