#include "ondrix/Dialect/ondsp/IR/OndspSemantics.h"

#include "ondrix/Dialect/ondsp/IR/OndspDialect.h"

#include "llvm/Support/raw_ostream.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"

using ondrix::ondsp::AccType;
using ondrix::ondsp::classifyReductionReassociation;
using ondrix::ondsp::FixedAttr;
using ondrix::ondsp::inferProductSemantics;
using ondrix::ondsp::isFullProduct;
using ondrix::ondsp::isRawHighProduct;
using ondrix::ondsp::isSignedI40Frac30Accumulator;
using ondrix::ondsp::isSignedI64Frac62Accumulator;
using ondrix::ondsp::isSignedQ15;
using ondrix::ondsp::isSignedQ31;
using ondrix::ondsp::OverflowMode;
using ondrix::ondsp::ProductAttr;
using ondrix::ondsp::ProductSelection;
using ondrix::ondsp::ProductSemantics;
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
  auto i64 = mlir::IntegerType::get(&context, 64);

  bool passed = true;
  passed &= isSignedQ15(FixedAttr::get(&context, Signedness::Signed, i16, 15));
  passed &= !isSignedQ15(FixedAttr::get(&context, Signedness::Unsigned, i16, 15));
  passed &= !isSignedQ15(FixedAttr::get(&context, Signedness::Signed, i32, 15));
  passed &= isSignedQ31(FixedAttr::get(&context, Signedness::Signed, i32, 31));
  passed &= !isSignedQ31(FixedAttr::get(&context, Signedness::Signed, i32, 30));
  passed &= isSignedI40Frac30Accumulator(
      AccType::get(&context, i40, 30, Signedness::Signed, OverflowMode::Saturate));
  passed &= isSignedI40Frac30Accumulator(
      AccType::get(&context, i40, 30, Signedness::Signed, OverflowMode::Wrap));
  passed &= !isSignedI40Frac30Accumulator(
      AccType::get(&context, i32, 30, Signedness::Signed, OverflowMode::Saturate));
  passed &= !isSignedI40Frac30Accumulator(
      AccType::get(&context, i40, 29, Signedness::Signed, OverflowMode::Saturate));
  passed &= isSignedI64Frac62Accumulator(
      AccType::get(&context, i64, 62, Signedness::Signed, OverflowMode::Saturate));
  passed &= !isSignedI64Frac62Accumulator(
      AccType::get(&context, i64, 61, Signedness::Signed, OverflowMode::Saturate));
  passed &= isFullProduct(ProductAttr::get(&context, ProductSelection::Full));
  passed &= !isFullProduct(ProductAttr::get(&context, ProductSelection::HighRaw));
  passed &= isRawHighProduct(ProductAttr::get(&context, ProductSelection::HighRaw));
  return passed;
}

bool testProductSemantics() {
  mlir::MLIRContext context;
  context.getOrLoadDialect<ondrix::ondsp::OndspDialect>();
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::ModuleOp::create(mlir::UnknownLoc::get(&context));
  auto i16 = mlir::IntegerType::get(&context, 16);
  auto i32 = mlir::IntegerType::get(&context, 32);

  auto q15 = FixedAttr::get(&context, Signedness::Signed, i16, 15);
  mlir::FailureOr<ProductSemantics> q15Full = inferProductSemantics(
      module->getOperation(), q15, ProductAttr::get(&context, ProductSelection::Full));
  if (mlir::failed(q15Full))
    return false;

  auto q31 = FixedAttr::get(&context, Signedness::Signed, i32, 31);
  mlir::FailureOr<ProductSemantics> q31Full = inferProductSemantics(
      module->getOperation(), q31, ProductAttr::get(&context, ProductSelection::Full));
  mlir::FailureOr<ProductSemantics> q31HighRaw = inferProductSemantics(
      module->getOperation(), q31, ProductAttr::get(&context, ProductSelection::HighRaw));
  if (mlir::failed(q31Full) || mlir::failed(q31HighRaw))
    return false;

  return q15Full->rawWidth == 32 && q15Full->frac == 30 &&
         q15Full->selection == ProductSelection::Full && q31Full->rawWidth == 64 &&
         q31Full->frac == 62 && q31Full->selection == ProductSelection::Full &&
         q31HighRaw->rawWidth == 32 && q31HighRaw->frac == 30 &&
         q31HighRaw->selection == ProductSelection::HighRaw;
}

} // namespace

int main() {
  if (!testReductionReassociationSafety() || !testCommonNumericPolicies() ||
      !testProductSemantics()) {
    llvm::errs() << "ondsp reassociation semantics: FAIL\n";
    return 1;
  }
  llvm::outs() << "ondsp reassociation semantics: PASS\n";
  return 0;
}
