#include "ondrix/Analysis/FixedPointPrefixRangeAnalysis.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/MathExtras.h"

#include "mlir/IR/BuiltinTypes.h"

#include <algorithm>
#include <limits>

using namespace mlir;

namespace ondrix::analysis {
namespace {

bool isValidInterval(const FixedPointRawInterval &interval) {
  return interval.lower.getBitWidth() == interval.upper.getBitWidth() &&
         interval.lower.sle(interval.upper);
}

bool hasFractionalPosition(ArrayRef<FixedPointRawInterval> intervals, unsigned frac) {
  return llvm::all_of(
      intervals, [frac](const FixedPointRawInterval &interval) { return interval.frac == frac; });
}

FailureOr<unsigned> getAnalysisWidth(unsigned accumulatorWidth,
                                     const FixedPointRawInterval &initial,
                                     ArrayRef<FixedPointRawInterval> originalUpdates,
                                     ArrayRef<FixedPointRawInterval> reassociatedUpdates) {
  unsigned maxValueWidth = std::max(accumulatorWidth, initial.lower.getBitWidth());
  auto includeInterval = [&](const FixedPointRawInterval &interval) {
    maxValueWidth = std::max(maxValueWidth, interval.lower.getBitWidth());
  };
  llvm::for_each(originalUpdates, includeInterval);
  llvm::for_each(reassociatedUpdates, includeInterval);

  uint64_t maxUpdateCount = std::max(originalUpdates.size(), reassociatedUpdates.size());
  if (maxUpdateCount == std::numeric_limits<uint64_t>::max())
    return failure();

  // One extra sign bit plus enough carry bits prevents the interval sums from
  // wrapping in APInt before they are compared with the accumulator bounds.
  unsigned growth = llvm::Log2_64_Ceil(maxUpdateCount + 1) + 1;
  if (maxValueWidth > std::numeric_limits<unsigned>::max() - growth)
    return failure();
  return maxValueWidth + growth;
}

FixedPointRawInterval extendInterval(const FixedPointRawInterval &interval, unsigned width) {
  return {interval.lower.sext(width), interval.upper.sext(width), interval.frac};
}

bool everyPrefixFits(unsigned accumulatorWidth, unsigned analysisWidth,
                     const FixedPointRawInterval &initial,
                     ArrayRef<FixedPointRawInterval> updates) {
  FixedPointRawInterval prefix = extendInterval(initial, analysisWidth);
  APInt accumulatorMinimum = APInt::getSignedMinValue(accumulatorWidth).sext(analysisWidth);
  APInt accumulatorMaximum = APInt::getSignedMaxValue(accumulatorWidth).sext(analysisWidth);
  auto fits = [&](const FixedPointRawInterval &interval) {
    return !interval.lower.slt(accumulatorMinimum) && !interval.upper.sgt(accumulatorMaximum);
  };
  if (!fits(prefix))
    return false;

  for (const FixedPointRawInterval &update : updates) {
    FixedPointRawInterval extended = extendInterval(update, analysisWidth);
    prefix.lower += extended.lower;
    prefix.upper += extended.upper;
    if (!fits(prefix))
      return false;
  }
  return true;
}

bool equalInterval(const FixedPointRawInterval &lhs, const FixedPointRawInterval &rhs) {
  return lhs.lower == rhs.lower && lhs.upper == rhs.upper && lhs.frac == rhs.frac;
}

bool equalIntervals(ArrayRef<FixedPointRawInterval> lhs, ArrayRef<FixedPointRawInterval> rhs) {
  return lhs.size() == rhs.size() && std::equal(lhs.begin(), lhs.end(), rhs.begin(), equalInterval);
}

bool equalCoefficientPair(const CoefficientPair &lhs, const CoefficientPair &rhs) {
  return lhs.originalLhsIndex == rhs.originalLhsIndex &&
         lhs.originalRhsIndex == rhs.originalRhsIndex &&
         lhs.reassociatedIndex == rhs.reassociatedIndex &&
         lhs.lhsCoefficient == rhs.lhsCoefficient && lhs.rhsCoefficient == rhs.rhsCoefficient;
}

bool equalCoefficientPairs(ArrayRef<CoefficientPair> lhs, ArrayRef<CoefficientPair> rhs) {
  return lhs.size() == rhs.size() &&
         std::equal(lhs.begin(), lhs.end(), rhs.begin(), equalCoefficientPair);
}

bool equalPassthroughUpdate(const PassthroughUpdate &lhs, const PassthroughUpdate &rhs) {
  return lhs.originalIndex == rhs.originalIndex && lhs.reassociatedIndex == rhs.reassociatedIndex;
}

bool equalPassthroughUpdates(ArrayRef<PassthroughUpdate> lhs, ArrayRef<PassthroughUpdate> rhs) {
  return lhs.size() == rhs.size() &&
         std::equal(lhs.begin(), lhs.end(), rhs.begin(), equalPassthroughUpdate);
}

bool intervalContains(const FixedPointRawInterval &container,
                      const FixedPointRawInterval &contained) {
  if (container.frac != contained.frac)
    return false;
  unsigned width = std::max(container.lower.getBitWidth(), contained.lower.getBitWidth());
  return container.lower.sext(width).sle(contained.lower.sext(width)) &&
         container.upper.sext(width).sge(contained.upper.sext(width));
}

FailureOr<FixedPointRawInterval> addIntervals(const FixedPointRawInterval &lhs,
                                              const FixedPointRawInterval &rhs) {
  if (lhs.frac != rhs.frac)
    return failure();
  unsigned inputWidth = std::max(lhs.lower.getBitWidth(), rhs.lower.getBitWidth());
  if (inputWidth == std::numeric_limits<unsigned>::max())
    return failure();
  unsigned width = inputWidth + 1;
  return FixedPointRawInterval{lhs.lower.sext(width) + rhs.lower.sext(width),
                               lhs.upper.sext(width) + rhs.upper.sext(width), lhs.frac};
}

bool isValidScheduleMapping(ArrayRef<CoefficientPair> pairs,
                            ArrayRef<PassthroughUpdate> passthroughs, unsigned coefficientWidth,
                            ArrayRef<FixedPointRawInterval> originalUpdates,
                            ArrayRef<FixedPointRawInterval> reassociatedUpdates) {
  if (pairs.empty() ||
      pairs.size() > (std::numeric_limits<size_t>::max() - passthroughs.size()) / 2 ||
      originalUpdates.size() != pairs.size() * 2 + passthroughs.size() ||
      reassociatedUpdates.size() != pairs.size() + passthroughs.size() ||
      originalUpdates.size() > static_cast<size_t>(std::numeric_limits<int64_t>::max()) ||
      reassociatedUpdates.size() > static_cast<size_t>(std::numeric_limits<int64_t>::max()))
    return false;

  llvm::SmallDenseSet<int64_t, 16> usedOriginalIndices;
  llvm::SmallDenseSet<int64_t, 16> usedReassociatedIndices;
  for (const CoefficientPair &pair : pairs) {
    if (pair.originalLhsIndex < 0 || pair.originalRhsIndex <= pair.originalLhsIndex ||
        pair.reassociatedIndex < 0 ||
        static_cast<size_t>(pair.originalRhsIndex) >= originalUpdates.size() ||
        static_cast<size_t>(pair.reassociatedIndex) >= reassociatedUpdates.size() ||
        pair.lhsCoefficient.getBitWidth() != coefficientWidth ||
        pair.rhsCoefficient.getBitWidth() != coefficientWidth ||
        pair.lhsCoefficient != pair.rhsCoefficient ||
        !usedOriginalIndices.insert(pair.originalLhsIndex).second ||
        !usedOriginalIndices.insert(pair.originalRhsIndex).second ||
        !usedReassociatedIndices.insert(pair.reassociatedIndex).second)
      return false;

    FailureOr<FixedPointRawInterval> pairedInterval = addIntervals(
        originalUpdates[pair.originalLhsIndex], originalUpdates[pair.originalRhsIndex]);
    if (failed(pairedInterval) ||
        !intervalContains(reassociatedUpdates[pair.reassociatedIndex], *pairedInterval))
      return false;
  }

  for (const PassthroughUpdate &passthrough : passthroughs) {
    if (passthrough.originalIndex < 0 || passthrough.reassociatedIndex < 0 ||
        static_cast<size_t>(passthrough.originalIndex) >= originalUpdates.size() ||
        static_cast<size_t>(passthrough.reassociatedIndex) >= reassociatedUpdates.size() ||
        !usedOriginalIndices.insert(passthrough.originalIndex).second ||
        !usedReassociatedIndices.insert(passthrough.reassociatedIndex).second ||
        !intervalContains(reassociatedUpdates[passthrough.reassociatedIndex],
                          originalUpdates[passthrough.originalIndex]))
      return false;
  }

  return usedOriginalIndices.size() == originalUpdates.size() &&
         usedReassociatedIndices.size() == reassociatedUpdates.size();
}

} // namespace

DistributivePairingPlan::DistributivePairingPlan(DistributivePairingPlan &&other)
    : subject(other.subject), semantics(std::move(other.semantics)), legality(other.legality),
      numeric(other.numeric), product(other.product), accumulator(other.accumulator),
      coefficientPairs(std::move(other.coefficientPairs)),
      passthroughUpdates(std::move(other.passthroughUpdates)), initial(std::move(other.initial)),
      originalUpdates(std::move(other.originalUpdates)),
      reassociatedUpdates(std::move(other.reassociatedUpdates)), consumed(other.consumed) {
  other.subject = nullptr;
  other.consumed = true;
}

DistributivePairingPlan &DistributivePairingPlan::operator=(DistributivePairingPlan &&other) {
  if (this == &other)
    return *this;
  subject = other.subject;
  semantics = std::move(other.semantics);
  legality = other.legality;
  numeric = other.numeric;
  product = other.product;
  accumulator = other.accumulator;
  coefficientPairs = std::move(other.coefficientPairs);
  passthroughUpdates = std::move(other.passthroughUpdates);
  initial = std::move(other.initial);
  originalUpdates = std::move(other.originalUpdates);
  reassociatedUpdates = std::move(other.reassociatedUpdates);
  consumed = other.consumed;
  other.subject = nullptr;
  other.consumed = true;
  return *this;
}

LogicalResult DistributivePairingPlan::consumeIfValid(
    Operation *operation, ondsp::FixedAttr candidateNumeric, ondsp::ProductAttr candidateProduct,
    ondsp::AccType candidateAccumulator, ArrayRef<CoefficientPair> candidateCoefficientPairs,
    ArrayRef<PassthroughUpdate> candidatePassthroughUpdates,
    const FixedPointRawInterval &candidateInitial,
    ArrayRef<FixedPointRawInterval> candidateOriginalUpdates,
    ArrayRef<FixedPointRawInterval> candidateReassociatedUpdates,
    DistributivePairingConsumer consumer) && {
  if (consumed)
    return failure();
  consumed = true;
  if (subject != operation || numeric != candidateNumeric || product != candidateProduct ||
      accumulator != candidateAccumulator ||
      !equalCoefficientPairs(coefficientPairs, candidateCoefficientPairs) ||
      !equalPassthroughUpdates(passthroughUpdates, candidatePassthroughUpdates) ||
      !equalInterval(initial, candidateInitial) ||
      !equalIntervals(originalUpdates, candidateOriginalUpdates) ||
      !equalIntervals(reassociatedUpdates, candidateReassociatedUpdates))
    return failure();
  return consumer(semantics, legality);
}

LogicalResult FixedPointPrefixRangePlanner::proveAllPrefixesFit(
    unsigned accumulatorWidth, const FixedPointRawInterval &initial,
    ArrayRef<FixedPointRawInterval> originalUpdates,
    ArrayRef<FixedPointRawInterval> reassociatedUpdates) {
  if (accumulatorWidth == 0 || !isValidInterval(initial) ||
      llvm::any_of(
          originalUpdates,
          [](const FixedPointRawInterval &interval) { return !isValidInterval(interval); }) ||
      llvm::any_of(
          reassociatedUpdates,
          [](const FixedPointRawInterval &interval) { return !isValidInterval(interval); }) ||
      !hasFractionalPosition(originalUpdates, initial.frac) ||
      !hasFractionalPosition(reassociatedUpdates, initial.frac))
    return failure();

  FailureOr<unsigned> analysisWidth =
      getAnalysisWidth(accumulatorWidth, initial, originalUpdates, reassociatedUpdates);
  if (failed(analysisWidth) ||
      !everyPrefixFits(accumulatorWidth, *analysisWidth, initial, originalUpdates) ||
      !everyPrefixFits(accumulatorWidth, *analysisWidth, initial, reassociatedUpdates))
    return failure();
  return success();
}

FailureOr<DistributivePairingPlan> FixedPointPrefixRangePlanner::planDistributivePairing(
    Operation *subject, ondsp::FixedAttr numeric, ondsp::ProductAttr product,
    ondsp::AccType accumulator, ArrayRef<CoefficientPair> coefficientPairs,
    ArrayRef<PassthroughUpdate> passthroughUpdates, const FixedPointRawInterval &initial,
    ArrayRef<FixedPointRawInterval> originalUpdates,
    ArrayRef<FixedPointRawInterval> reassociatedUpdates) {
  auto numericStorage = dyn_cast<IntegerType>(numeric.getStorage());
  auto accumulatorStorage = dyn_cast<IntegerType>(accumulator.getStorage());
  if (!subject || !numericStorage || !accumulatorStorage || !isValidInterval(initial) ||
      initial.frac != accumulator.getFrac() ||
      !hasFractionalPosition(originalUpdates, accumulator.getFrac()) ||
      !hasFractionalPosition(reassociatedUpdates, accumulator.getFrac()) ||
      llvm::any_of(
          originalUpdates,
          [](const FixedPointRawInterval &interval) { return !isValidInterval(interval); }) ||
      llvm::any_of(
          reassociatedUpdates,
          [](const FixedPointRawInterval &interval) { return !isValidInterval(interval); }) ||
      !isValidScheduleMapping(coefficientPairs, passthroughUpdates, numericStorage.getWidth(),
                              originalUpdates, reassociatedUpdates))
    return failure();

  ArrayRef<FixedPointRawInterval> noUpdates;
  if (failed(proveAllPrefixesFit(accumulatorStorage.getWidth(), initial, noUpdates, noUpdates)))
    return failure();

  FailureOr<ondsp::DistributivePairingSemantics> semantics =
      ondsp::classifyDistributiveProductPairing(subject, numeric, product, accumulator);
  if (failed(semantics) || !semantics->exactBeforeAccumulatorOverflow)
    return failure();

  ondsp::TransformLegality legality = semantics->legalityWithoutRangeProof;
  if (!legality.isExact()) {
    if (accumulator.getUpdateOverflow() != ondsp::OverflowMode::Saturate ||
        failed(proveAllPrefixesFit(accumulatorStorage.getWidth(), initial, originalUpdates,
                                   reassociatedUpdates)))
      return failure();
    legality = ondsp::TransformLegality::getNoOverflowProof();
  }

  return DistributivePairingPlan(subject, std::move(*semantics), legality, numeric, product,
                                 accumulator, coefficientPairs, passthroughUpdates, initial,
                                 originalUpdates, reassociatedUpdates);
}

} // namespace ondrix::analysis
