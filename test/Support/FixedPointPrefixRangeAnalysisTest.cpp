#include "ondrix/Analysis/FixedPointPrefixRangeAnalysis.h"

#include "ondrix/Dialect/ondsp/IR/OndspDialect.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"

#include <cstdint>
#include <type_traits>

using ondrix::analysis::addFixedPointRawIntervals;
using ondrix::analysis::computeSignedFullProductInterval;
using ondrix::analysis::DistributivePairingPlan;
using ondrix::analysis::FixedPointPrefixRangePlanner;
using ondrix::analysis::FixedPointRawInterval;
using ondrix::ondsp::AccType;
using ondrix::ondsp::FixedAttr;
using ondrix::ondsp::OverflowMode;
using ondrix::ondsp::ProductAttr;
using ondrix::ondsp::ProductSelection;
using ondrix::ondsp::Signedness;
using ondrix::ondsp::TransformJustification;

namespace {

llvm::APInt signedValue(unsigned width, int64_t value) {
  return llvm::APInt(width, static_cast<uint64_t>(value), true);
}

FixedPointRawInterval range(unsigned width, int64_t lower, int64_t upper, unsigned frac = 0) {
  return {signedValue(width, lower), signedValue(width, upper), frac};
}

FixedPointRawInterval exactRange(const llvm::APInt &value, unsigned frac) {
  return {value, value, frac};
}

bool equals(const FixedPointRawInterval &interval, const FixedPointRawInterval &expected) {
  return interval.lower == expected.lower && interval.upper == expected.upper &&
         interval.frac == expected.frac;
}

static_assert(!std::is_copy_constructible_v<DistributivePairingPlan>);
static_assert(!std::is_copy_assignable_v<DistributivePairingPlan>);
static_assert(std::is_move_constructible_v<DistributivePairingPlan>);
static_assert(std::is_move_assignable_v<DistributivePairingPlan>);

bool testSignedFullProductIntervals() {
  mlir::MLIRContext context;
  context.getOrLoadDialect<ondrix::ondsp::OndspDialect>();
  auto i16 = mlir::IntegerType::get(&context, 16);
  auto i32 = mlir::IntegerType::get(&context, 32);
  auto q15 = FixedAttr::get(&context, Signedness::Signed, i16, 15);
  auto q31 = FixedAttr::get(&context, Signedness::Signed, i32, 31);
  auto unsignedQ15 = FixedAttr::get(&context, Signedness::Unsigned, i16, 15);

  auto q15Maximum = computeSignedFullProductInterval(q15, signedValue(16, INT16_MAX));
  auto q15Minimum = computeSignedFullProductInterval(q15, signedValue(16, INT16_MIN));
  auto q15Zero = computeSignedFullProductInterval(q15, signedValue(16, 0));
  auto q31Minimum = computeSignedFullProductInterval(q31, signedValue(32, INT32_MIN));
  auto unsignedProduct = computeSignedFullProductInterval(unsignedQ15, signedValue(16, INT16_MAX));
  auto wrongCoefficientWidth = computeSignedFullProductInterval(q15, signedValue(32, 1));

  return mlir::succeeded(q15Maximum) &&
         equals(*q15Maximum, range(32, INT64_C(-1073709056), INT64_C(1073676289), 30)) &&
         mlir::succeeded(q15Minimum) &&
         equals(*q15Minimum, range(32, INT64_C(-1073709056), INT64_C(1073741824), 30)) &&
         mlir::succeeded(q15Zero) && equals(*q15Zero, range(32, 0, 0, 30)) &&
         mlir::succeeded(q31Minimum) &&
         equals(*q31Minimum,
                range(64, INT64_C(-4611686016279904256), INT64_C(4611686018427387904), 62)) &&
         mlir::failed(unsignedProduct) && mlir::failed(wrongCoefficientWidth);
}

bool testRawIntervalAddition() {
  FixedPointRawInterval lhs = range(32, INT32_MIN, INT32_MAX, 30);
  FixedPointRawInterval rhs = range(32, INT32_MIN, INT32_MAX, 30);
  auto sum = addFixedPointRawIntervals(lhs, rhs);
  auto mismatchedFrac = addFixedPointRawIntervals(lhs, range(32, 0, 0, 29));
  auto invalid = addFixedPointRawIntervals(range(32, 1, -1, 30), rhs);
  return mlir::succeeded(sum) &&
         equals(*sum, range(33, INT64_C(-4294967296), INT64_C(4294967294), 30)) &&
         mlir::failed(mismatchedFrac) && mlir::failed(invalid);
}

bool testSuccessfulContainment() {
  FixedPointRawInterval initial = range(4, 0, 0);
  llvm::SmallVector<FixedPointRawInterval> original = {range(4, 1, 2), range(4, -1, 1)};
  llvm::SmallVector<FixedPointRawInterval> reassociated = {range(4, 0, 3)};
  return mlir::succeeded(
      FixedPointPrefixRangePlanner::proveAllPrefixesFit(4, initial, original, reassociated));
}

bool testOriginalPrefixOverflow() {
  FixedPointRawInterval initial = range(4, 0, 0);
  llvm::SmallVector<FixedPointRawInterval> original = {range(4, 7, 7), range(4, 1, 1),
                                                       range(4, -1, -1)};
  llvm::SmallVector<FixedPointRawInterval> reassociated = {range(4, 7, 7), range(4, 0, 0)};
  return mlir::failed(
      FixedPointPrefixRangePlanner::proveAllPrefixesFit(4, initial, original, reassociated));
}

bool testReassociatedPrefixOverflow() {
  FixedPointRawInterval initial = range(5, 0, 0);
  llvm::SmallVector<FixedPointRawInterval> original = {range(5, 7, 7), range(5, -7, -7)};
  llvm::SmallVector<FixedPointRawInterval> reassociated = {range(5, 8, 8), range(5, -8, -8)};
  return mlir::failed(
      FixedPointPrefixRangePlanner::proveAllPrefixesFit(4, initial, original, reassociated));
}

bool testNegativePrefixUnderflow() {
  FixedPointRawInterval initial = range(4, 0, 0);
  llvm::SmallVector<FixedPointRawInterval> original = {range(4, -8, -8), range(4, -1, -1),
                                                       range(4, 1, 1)};
  llvm::SmallVector<FixedPointRawInterval> reassociated = {range(4, -8, -8), range(4, 0, 0)};
  return mlir::failed(
      FixedPointPrefixRangePlanner::proveAllPrefixesFit(4, initial, original, reassociated));
}

bool testWiderAccumulator() {
  FixedPointRawInterval initial = range(40, 0, 0, 30);
  llvm::SmallVector<FixedPointRawInterval> original = {range(32, INT32_MIN, INT32_MAX, 30),
                                                       range(32, INT32_MIN, INT32_MAX, 30)};
  llvm::SmallVector<FixedPointRawInterval> reassociated = {
      range(33, INT64_C(-4294967296), INT64_C(4294967294), 30)};
  return mlir::succeeded(
      FixedPointPrefixRangePlanner::proveAllPrefixesFit(40, initial, original, reassociated));
}

bool testQ31I65Boundaries() {
  llvm::APInt positiveTwo62 = llvm::APInt::getOneBitSet(64, 62);
  llvm::APInt negativeTwo62 = -positiveTwo62;
  llvm::APInt positiveTwo63 = llvm::APInt::getOneBitSet(65, 63);
  llvm::APInt negativeTwo63 = -positiveTwo63;
  FixedPointRawInterval initial = range(64, 0, 0, 62);

  llvm::SmallVector<FixedPointRawInterval> original = {exactRange(positiveTwo62, 62),
                                                       exactRange(negativeTwo62, 62)};
  llvm::SmallVector<FixedPointRawInterval> reassociated = {range(65, 0, 0, 62)};
  llvm::SmallVector<FixedPointRawInterval> exactNegativeEndpoint = {exactRange(negativeTwo63, 62)};
  llvm::SmallVector<FixedPointRawInterval> positiveOverflow = {exactRange(positiveTwo63, 62)};

  return mlir::succeeded(FixedPointPrefixRangePlanner::proveAllPrefixesFit(64, initial, original,
                                                                           reassociated)) &&
         mlir::succeeded(FixedPointPrefixRangePlanner::proveAllPrefixesFit(
             64, initial, exactNegativeEndpoint, exactNegativeEndpoint)) &&
         mlir::failed(FixedPointPrefixRangePlanner::proveAllPrefixesFit(
             64, initial, positiveOverflow, positiveOverflow));
}

bool testPairingPlanValidation() {
  mlir::MLIRContext context;
  context.getOrLoadDialect<ondrix::ondsp::OndspDialect>();
  mlir::OwningOpRef<mlir::ModuleOp> subject =
      mlir::ModuleOp::create(mlir::UnknownLoc::get(&context));
  mlir::OwningOpRef<mlir::ModuleOp> other = mlir::ModuleOp::create(mlir::UnknownLoc::get(&context));
  auto i16 = mlir::IntegerType::get(&context, 16);
  auto i40 = mlir::IntegerType::get(&context, 40);
  auto numeric = FixedAttr::get(&context, Signedness::Signed, i16, 15);
  auto product = ProductAttr::get(&context, ProductSelection::Full);
  auto accumulator = AccType::get(&context, i40, 30, Signedness::Signed, OverflowMode::Saturate);
  llvm::SmallVector<llvm::APInt> coefficients = {signedValue(16, 3), signedValue(16, 3)};

  auto buildPlan = [&]() {
    return FixedPointPrefixRangePlanner::planZeroSeededSymmetricPairing(
        subject->getOperation(), numeric, product, accumulator, coefficients);
  };

  auto moveSource = buildPlan();
  if (mlir::failed(moveSource))
    return false;
  DistributivePairingPlan movedPlan = std::move(*moveSource);
  bool movedFromConsumed = false;
  if (mlir::succeeded(std::move(*moveSource)
                          .consumeIfValid(subject->getOperation(), numeric, product, accumulator,
                                          coefficients,
                                          [&](const auto &, const auto &) {
                                            movedFromConsumed = true;
                                            return mlir::success();
                                          })) ||
      movedFromConsumed)
    return false;
  bool movedPlanConsumed = false;
  if (mlir::failed(std::move(movedPlan).consumeIfValid(subject->getOperation(), numeric, product,
                                                       accumulator, coefficients,
                                                       [&](const auto &, const auto &) {
                                                         movedPlanConsumed = true;
                                                         return mlir::success();
                                                       })) ||
      !movedPlanConsumed)
    return false;

  auto plan = buildPlan();
  if (mlir::failed(plan))
    return false;

  bool consumed = false;
  auto consume = [&](const ondrix::ondsp::DistributivePairingSemantics &semantics,
                     llvm::ArrayRef<llvm::APInt> validatedCoefficients) {
    consumed = semantics.product.rawWidth == 32 && semantics.product.frac == 30 &&
               validatedCoefficients.size() == 2 && validatedCoefficients[0] == coefficients[0] &&
               validatedCoefficients[1] == coefficients[1];
    return consumed ? mlir::success() : mlir::failure();
  };
  if (mlir::failed(std::move(*plan).consumeIfValid(subject->getOperation(), numeric, product,
                                                   accumulator, coefficients, consume)) ||
      !consumed)
    return false;

  bool reused = false;
  if (mlir::succeeded(std::move(*plan).consumeIfValid(subject->getOperation(), numeric, product,
                                                      accumulator, coefficients,
                                                      [&](const auto &, const auto &) {
                                                        reused = true;
                                                        return mlir::success();
                                                      })) ||
      reused)
    return false;

  auto stalePlan = buildPlan();
  auto changedPlan = buildPlan();
  if (mlir::failed(stalePlan) || mlir::failed(changedPlan))
    return false;
  bool staleConsumed = false;
  llvm::SmallVector<llvm::APInt> changedCoefficients = {signedValue(16, 4), signedValue(16, 4)};
  return mlir::failed(std::move(*stalePlan)
                          .consumeIfValid(other->getOperation(), numeric, product, accumulator,
                                          coefficients,
                                          [&](const auto &, const auto &) {
                                            staleConsumed = true;
                                            return mlir::success();
                                          })) &&
         !staleConsumed &&
         mlir::failed(std::move(*changedPlan)
                          .consumeIfValid(subject->getOperation(), numeric, product, accumulator,
                                          changedCoefficients,
                                          [&](const auto &, const auto &) {
                                            staleConsumed = true;
                                            return mlir::success();
                                          })) &&
         !staleConsumed;
}

bool testWrappingPairingPlan() {
  mlir::MLIRContext context;
  context.getOrLoadDialect<ondrix::ondsp::OndspDialect>();
  mlir::OwningOpRef<mlir::ModuleOp> subject =
      mlir::ModuleOp::create(mlir::UnknownLoc::get(&context));
  auto i16 = mlir::IntegerType::get(&context, 16);
  auto i40 = mlir::IntegerType::get(&context, 40);
  auto numeric = FixedAttr::get(&context, Signedness::Signed, i16, 15);
  auto product = ProductAttr::get(&context, ProductSelection::Full);
  auto accumulator = AccType::get(&context, i40, 30, Signedness::Signed, OverflowMode::Wrap);
  llvm::SmallVector<llvm::APInt> coefficients(512, signedValue(16, INT16_MIN));

  auto plan = FixedPointPrefixRangePlanner::planZeroSeededSymmetricPairing(
      subject->getOperation(), numeric, product, accumulator, coefficients);
  if (mlir::failed(plan))
    return false;
  bool exactModulo = false;
  return mlir::succeeded(std::move(*plan).consumeIfValid(
             subject->getOperation(), numeric, product, accumulator, coefficients,
             [&](const ondrix::ondsp::DistributivePairingSemantics &semantics, const auto &) {
               exactModulo = semantics.legalityWithoutRangeProof.isExactWith(
                   TransformJustification::FixedWidthModulo);
               return exactModulo ? mlir::success() : mlir::failure();
             })) &&
         exactModulo;
}

bool testPlannerDerivesTrustedSchedules() {
  mlir::MLIRContext context;
  context.getOrLoadDialect<ondrix::ondsp::OndspDialect>();
  mlir::OwningOpRef<mlir::ModuleOp> subject =
      mlir::ModuleOp::create(mlir::UnknownLoc::get(&context));
  auto i16 = mlir::IntegerType::get(&context, 16);
  auto i40 = mlir::IntegerType::get(&context, 40);
  auto numeric = FixedAttr::get(&context, Signedness::Signed, i16, 15);
  auto product = ProductAttr::get(&context, ProductSelection::Full);
  auto saturating = AccType::get(&context, i40, 30, Signedness::Signed, OverflowMode::Saturate);
  auto wrapping = AccType::get(&context, i40, 30, Signedness::Signed, OverflowMode::Wrap);
  llvm::SmallVector<llvm::APInt> safeOdd = {signedValue(16, 1), signedValue(16, 2),
                                            signedValue(16, 1)};
  llvm::SmallVector<llvm::APInt> nonsymmetric = {signedValue(16, 1), signedValue(16, 2)};
  llvm::SmallVector<llvm::APInt> wrongWidth = {signedValue(32, 1), signedValue(32, 1)};
  llvm::SmallVector<llvm::APInt> oneCoefficient = {signedValue(16, 1)};
  llvm::SmallVector<llvm::APInt> overflowing(512, signedValue(16, INT16_MIN));

  auto plan = [&](AccType accumulator, llvm::ArrayRef<llvm::APInt> coefficients) {
    return FixedPointPrefixRangePlanner::planZeroSeededSymmetricPairing(
        subject->getOperation(), numeric, product, accumulator, coefficients);
  };
  return mlir::succeeded(plan(saturating, safeOdd)) &&
         mlir::failed(plan(saturating, nonsymmetric)) &&
         mlir::failed(plan(saturating, wrongWidth)) &&
         mlir::failed(plan(saturating, oneCoefficient)) &&
         mlir::failed(plan(saturating, overflowing)) &&
         mlir::succeeded(plan(wrapping, overflowing));
}

bool testInvalidInputs() {
  llvm::SmallVector<FixedPointRawInterval> noUpdates;
  llvm::SmallVector<FixedPointRawInterval> differentFrac = {range(4, 0, 0, 1)};
  return mlir::failed(FixedPointPrefixRangePlanner::proveAllPrefixesFit(0, range(4, 0, 0),
                                                                        noUpdates, noUpdates)) &&
         mlir::failed(FixedPointPrefixRangePlanner::proveAllPrefixesFit(4, range(4, 1, -1),
                                                                        noUpdates, noUpdates)) &&
         mlir::failed(FixedPointPrefixRangePlanner::proveAllPrefixesFit(4, range(5, 8, 8),
                                                                        noUpdates, noUpdates)) &&
         mlir::failed(FixedPointPrefixRangePlanner::proveAllPrefixesFit(4, range(4, 0, 0),
                                                                        differentFrac, noUpdates));
}

} // namespace

int main() {
  if (!testSignedFullProductIntervals() || !testRawIntervalAddition() ||
      !testSuccessfulContainment() || !testOriginalPrefixOverflow() ||
      !testReassociatedPrefixOverflow() || !testNegativePrefixUnderflow() ||
      !testWiderAccumulator() || !testQ31I65Boundaries() || !testPairingPlanValidation() ||
      !testWrappingPairingPlan() || !testPlannerDerivesTrustedSchedules() || !testInvalidInputs()) {
    llvm::errs() << "fixed-point prefix range analysis: FAIL\n";
    return 1;
  }
  llvm::outs() << "fixed-point prefix range analysis: PASS\n";
  return 0;
}
