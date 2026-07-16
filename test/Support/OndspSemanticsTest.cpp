#include "ondrix/Dialect/ondsp/IR/OndspSemantics.h"

#include "ondrix/Dialect/ondsp/IR/OndspDialect.h"

#include "llvm/Support/raw_ostream.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"

using ondrix::ondsp::AccType;
using ondrix::ondsp::classifyReductionReassociation;
using ondrix::ondsp::FixedAttr;
using ondrix::ondsp::isFullProduct;
using ondrix::ondsp::isSignedQ15;
using ondrix::ondsp::isSignedQ15I40Accumulator;
using ondrix::ondsp::OverflowMode;
using ondrix::ondsp::ProductAttr;
using ondrix::ondsp::ProductSelection;
using ondrix::ondsp::ReductionRangeProof;
using ondrix::ondsp::ReductionReassociationSafety;
using ondrix::ondsp::Signedness;

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

bool testCommonNumericPolicies() {
  mlir::MLIRContext context;
  context.getOrLoadDialect<ondrix::ondsp::OndspDialect>();
  auto i16 = mlir::IntegerType::get(&context, 16);
  auto i32 = mlir::IntegerType::get(&context, 32);
  auto i40 = mlir::IntegerType::get(&context, 40);

  bool passed = true;
  passed &= isSignedQ15(FixedAttr::get(&context, Signedness::Signed, i16, 15));
  passed &= !isSignedQ15(FixedAttr::get(&context, Signedness::Unsigned, i16, 15));
  passed &= !isSignedQ15(FixedAttr::get(&context, Signedness::Signed, i32, 15));
  passed &= isSignedQ15I40Accumulator(
      AccType::get(&context, i40, 30, Signedness::Signed, OverflowMode::Saturate));
  passed &= isSignedQ15I40Accumulator(
      AccType::get(&context, i40, 30, Signedness::Signed, OverflowMode::Wrap));
  passed &= !isSignedQ15I40Accumulator(
      AccType::get(&context, i32, 30, Signedness::Signed, OverflowMode::Saturate));
  passed &= !isSignedQ15I40Accumulator(
      AccType::get(&context, i40, 29, Signedness::Signed, OverflowMode::Saturate));
  passed &= isFullProduct(ProductAttr::get(&context, ProductSelection::Full));
  passed &= !isFullProduct(ProductAttr::get(&context, ProductSelection::High));
  return passed;
}

} // namespace

int main() {
  if (!testReductionReassociationSafety() || !testCommonNumericPolicies()) {
    llvm::errs() << "ondsp reassociation semantics: FAIL\n";
    return 1;
  }
  llvm::outs() << "ondsp reassociation semantics: PASS\n";
  return 0;
}
