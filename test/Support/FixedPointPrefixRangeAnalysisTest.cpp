#include "ondrix/Analysis/FixedPointPrefixRangeAnalysis.h"

#include "ondrix/Dialect/ondsp/IR/OndspDialect.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"

#include <cstdint>
#include <type_traits>

using ondrix::analysis::CoefficientPair;
using ondrix::analysis::DistributivePairingPlan;
using ondrix::analysis::FixedPointPrefixRangePlanner;
using ondrix::analysis::FixedPointRawInterval;
using ondrix::analysis::PassthroughUpdate;
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

CoefficientPair coefficientPair(int64_t lhsIndex, int64_t rhsIndex, int64_t reassociatedIndex,
                                int64_t lhsValue, int64_t rhsValue) {
  return {lhsIndex, rhsIndex, reassociatedIndex, signedValue(16, lhsValue),
          signedValue(16, rhsValue)};
}

static_assert(!std::is_copy_constructible_v<DistributivePairingPlan>);
static_assert(!std::is_copy_assignable_v<DistributivePairingPlan>);
static_assert(std::is_move_constructible_v<DistributivePairingPlan>);
static_assert(std::is_move_assignable_v<DistributivePairingPlan>);

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
  llvm::SmallVector<CoefficientPair> pairs = {coefficientPair(0, 1, 0, 3, 3)};
  llvm::SmallVector<PassthroughUpdate> passthroughs;
  FixedPointRawInterval initial = range(40, 0, 0, 30);
  llvm::SmallVector<FixedPointRawInterval> original = {range(32, -16, 16, 30),
                                                       range(32, -16, 16, 30)};
  llvm::SmallVector<FixedPointRawInterval> reassociated = {range(33, -32, 32, 30)};

  auto buildPlan = [&]() {
    return FixedPointPrefixRangePlanner::planDistributivePairing(
        subject->getOperation(), numeric, product, accumulator, pairs, passthroughs, initial,
        original, reassociated);
  };

  auto moveSource = buildPlan();
  if (mlir::failed(moveSource))
    return false;
  DistributivePairingPlan movedPlan = std::move(*moveSource);
  bool movedFromConsumed = false;
  if (mlir::succeeded(std::move(*moveSource)
                          .consumeIfValid(subject->getOperation(), numeric, product, accumulator,
                                          pairs, passthroughs, initial, original, reassociated,
                                          [&](const auto &, const auto &) {
                                            movedFromConsumed = true;
                                            return mlir::success();
                                          })) ||
      movedFromConsumed)
    return false;
  bool movedPlanConsumed = false;
  if (mlir::failed(std::move(movedPlan).consumeIfValid(subject->getOperation(), numeric, product,
                                                       accumulator, pairs, passthroughs, initial,
                                                       original, reassociated,
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
                     const ondrix::ondsp::TransformLegality &legality) {
    consumed = legality.isExactWith(TransformJustification::NoOverflowProof) &&
               semantics.product.rawWidth == 32 && semantics.product.frac == 30;
    return consumed ? mlir::success() : mlir::failure();
  };
  if (mlir::failed(std::move(*plan).consumeIfValid(subject->getOperation(), numeric, product,
                                                   accumulator, pairs, passthroughs, initial,
                                                   original, reassociated, consume)) ||
      !consumed)
    return false;

  bool reused = false;
  if (mlir::succeeded(std::move(*plan).consumeIfValid(subject->getOperation(), numeric, product,
                                                      accumulator, pairs, passthroughs, initial,
                                                      original, reassociated,
                                                      [&](const auto &, const auto &) {
                                                        reused = true;
                                                        return mlir::success();
                                                      })) ||
      reused)
    return false;

  auto stalePlan = buildPlan();
  if (mlir::failed(stalePlan))
    return false;
  bool staleConsumed = false;
  return mlir::failed(std::move(*stalePlan)
                          .consumeIfValid(other->getOperation(), numeric, product, accumulator,
                                          pairs, passthroughs, initial, original, reassociated,
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
  llvm::SmallVector<CoefficientPair> pairs = {coefficientPair(0, 1, 0, 3, 3)};
  llvm::SmallVector<PassthroughUpdate> passthroughs;
  FixedPointRawInterval initial = range(40, INT64_C(549755813887), INT64_C(549755813887), 30);
  llvm::SmallVector<FixedPointRawInterval> original = {range(32, 1, 1, 30), range(32, 1, 1, 30)};
  llvm::SmallVector<FixedPointRawInterval> reassociated = {range(33, 2, 2, 30)};

  auto plan = FixedPointPrefixRangePlanner::planDistributivePairing(
      subject->getOperation(), numeric, product, accumulator, pairs, passthroughs, initial,
      original, reassociated);
  if (mlir::failed(plan))
    return false;
  bool exactModulo = false;
  return mlir::succeeded(std::move(*plan).consumeIfValid(
             subject->getOperation(), numeric, product, accumulator, pairs, passthroughs, initial,
             original, reassociated,
             [&](const auto &, const ondrix::ondsp::TransformLegality &legality) {
               exactModulo = legality.isExactWith(TransformJustification::FixedWidthModulo);
               return exactModulo ? mlir::success() : mlir::failure();
             })) &&
         exactModulo;
}

bool testCompleteScheduleMapping() {
  mlir::MLIRContext context;
  context.getOrLoadDialect<ondrix::ondsp::OndspDialect>();
  mlir::OwningOpRef<mlir::ModuleOp> subject =
      mlir::ModuleOp::create(mlir::UnknownLoc::get(&context));
  auto i16 = mlir::IntegerType::get(&context, 16);
  auto i40 = mlir::IntegerType::get(&context, 40);
  auto numeric = FixedAttr::get(&context, Signedness::Signed, i16, 15);
  auto product = ProductAttr::get(&context, ProductSelection::Full);
  auto accumulator = AccType::get(&context, i40, 30, Signedness::Signed, OverflowMode::Saturate);
  llvm::SmallVector<CoefficientPair> pairs = {coefficientPair(0, 1, 0, 3, 3)};
  llvm::SmallVector<PassthroughUpdate> passthroughs = {{2, 1}};
  FixedPointRawInterval initial = range(40, 0, 0, 30);
  llvm::SmallVector<FixedPointRawInterval> original = {range(32, 1, 1, 30), range(32, 1, 1, 30),
                                                       range(32, 1, 1, 30)};
  llvm::SmallVector<FixedPointRawInterval> validCandidate = {range(33, 2, 2, 30),
                                                             range(32, 1, 1, 30)};
  llvm::SmallVector<FixedPointRawInterval> wrongPassthrough = {range(33, 2, 2, 30),
                                                               range(32, 2, 2, 30)};
  llvm::SmallVector<FixedPointRawInterval> narrowPair = {range(33, 1, 1, 30), range(32, 1, 1, 30)};
  llvm::SmallVector<PassthroughUpdate> missingPassthrough;

  auto plan = FixedPointPrefixRangePlanner::planDistributivePairing(
      subject->getOperation(), numeric, product, accumulator, pairs, passthroughs, initial,
      original, validCandidate);
  auto wrongPassthroughPlan = FixedPointPrefixRangePlanner::planDistributivePairing(
      subject->getOperation(), numeric, product, accumulator, pairs, passthroughs, initial,
      original, wrongPassthrough);
  auto narrowPairPlan = FixedPointPrefixRangePlanner::planDistributivePairing(
      subject->getOperation(), numeric, product, accumulator, pairs, passthroughs, initial,
      original, narrowPair);
  auto incompletePlan = FixedPointPrefixRangePlanner::planDistributivePairing(
      subject->getOperation(), numeric, product, accumulator, pairs, missingPassthrough, initial,
      original, validCandidate);
  return mlir::succeeded(plan) && mlir::failed(wrongPassthroughPlan) &&
         mlir::failed(narrowPairPlan) && mlir::failed(incompletePlan);
}

bool testInvalidPairingSchedules() {
  mlir::MLIRContext context;
  context.getOrLoadDialect<ondrix::ondsp::OndspDialect>();
  mlir::OwningOpRef<mlir::ModuleOp> subject =
      mlir::ModuleOp::create(mlir::UnknownLoc::get(&context));
  auto i16 = mlir::IntegerType::get(&context, 16);
  auto i40 = mlir::IntegerType::get(&context, 40);
  auto numeric = FixedAttr::get(&context, Signedness::Signed, i16, 15);
  auto product = ProductAttr::get(&context, ProductSelection::Full);
  auto accumulator = AccType::get(&context, i40, 30, Signedness::Signed, OverflowMode::Saturate);
  FixedPointRawInterval initial = range(40, 0, 0, 30);
  llvm::SmallVector<FixedPointRawInterval> original = {range(32, 0, 0, 30), range(32, 0, 0, 30),
                                                       range(32, 0, 0, 30), range(32, 0, 0, 30)};
  llvm::SmallVector<FixedPointRawInterval> reassociated = {range(33, 0, 0, 30),
                                                           range(33, 0, 0, 30)};
  llvm::SmallVector<PassthroughUpdate> passthroughs;
  llvm::SmallVector<CoefficientPair> duplicateOriginal = {coefficientPair(0, 1, 0, 3, 3),
                                                          coefficientPair(1, 2, 1, 3, 3)};
  llvm::SmallVector<CoefficientPair> duplicateReassociated = {coefficientPair(0, 1, 0, 3, 3),
                                                              coefficientPair(2, 3, 0, 3, 3)};
  llvm::SmallVector<CoefficientPair> reversed = {coefficientPair(1, 0, 0, 3, 3),
                                                 coefficientPair(2, 3, 1, 3, 3)};
  llvm::SmallVector<CoefficientPair> outOfRange = {coefficientPair(0, 4, 0, 3, 3),
                                                   coefficientPair(2, 3, 1, 3, 3)};

  auto fails = [&](llvm::ArrayRef<CoefficientPair> candidatePairs) {
    return mlir::failed(FixedPointPrefixRangePlanner::planDistributivePairing(
        subject->getOperation(), numeric, product, accumulator, candidatePairs, passthroughs,
        initial, original, reassociated));
  };
  return fails(duplicateOriginal) && fails(duplicateReassociated) && fails(reversed) &&
         fails(outOfRange);
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
  if (!testSuccessfulContainment() || !testOriginalPrefixOverflow() ||
      !testReassociatedPrefixOverflow() || !testNegativePrefixUnderflow() ||
      !testWiderAccumulator() || !testQ31I65Boundaries() || !testPairingPlanValidation() ||
      !testWrappingPairingPlan() || !testCompleteScheduleMapping() ||
      !testInvalidPairingSchedules() || !testInvalidInputs()) {
    llvm::errs() << "fixed-point prefix range analysis: FAIL\n";
    return 1;
  }
  llvm::outs() << "fixed-point prefix range analysis: PASS\n";
  return 0;
}
