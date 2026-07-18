#include "ondrix/Analysis/FixedPointPrefixRangeAnalysis.h"

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

bool equalCoefficients(ArrayRef<APInt> lhs, ArrayRef<APInt> rhs) {
  return lhs.size() == rhs.size() && std::equal(lhs.begin(), lhs.end(), rhs.begin());
}

} // namespace

FailureOr<FixedPointRawInterval> computeSignedFullProductInterval(ondsp::FixedAttr numeric,
                                                                  const APInt &coefficient) {
  auto storage = dyn_cast<IntegerType>(numeric.getStorage());
  if (!storage || !storage.isSignless() || numeric.getSignedness() != ondsp::Signedness::Signed ||
      coefficient.getBitWidth() != storage.getWidth() ||
      storage.getWidth() > std::numeric_limits<unsigned>::max() / 2 ||
      numeric.getFrac() > std::numeric_limits<unsigned>::max() / 2)
    return failure();

  unsigned productWidth = storage.getWidth() * 2;
  unsigned productFrac = numeric.getFrac() * 2;
  APInt coefficientExtended = coefficient.sext(productWidth);
  APInt first =
      APInt::getSignedMinValue(storage.getWidth()).sext(productWidth) * coefficientExtended;
  APInt second =
      APInt::getSignedMaxValue(storage.getWidth()).sext(productWidth) * coefficientExtended;
  if (first.sle(second))
    return FixedPointRawInterval{std::move(first), std::move(second), productFrac};
  return FixedPointRawInterval{std::move(second), std::move(first), productFrac};
}

FailureOr<FixedPointRawInterval> addFixedPointRawIntervals(const FixedPointRawInterval &lhs,
                                                           const FixedPointRawInterval &rhs) {
  if (!isValidInterval(lhs) || !isValidInterval(rhs) || lhs.frac != rhs.frac)
    return failure();
  unsigned inputWidth = std::max(lhs.lower.getBitWidth(), rhs.lower.getBitWidth());
  if (inputWidth == std::numeric_limits<unsigned>::max())
    return failure();
  unsigned width = inputWidth + 1;
  return FixedPointRawInterval{lhs.lower.sext(width) + rhs.lower.sext(width),
                               lhs.upper.sext(width) + rhs.upper.sext(width), lhs.frac};
}

DistributivePairingPlan::DistributivePairingPlan(DistributivePairingPlan &&other)
    : subject(other.subject), semantics(std::move(other.semantics)), numeric(other.numeric),
      product(other.product), accumulator(other.accumulator),
      coefficients(std::move(other.coefficients)), consumed(other.consumed) {
  other.subject = nullptr;
  other.consumed = true;
}

DistributivePairingPlan &DistributivePairingPlan::operator=(DistributivePairingPlan &&other) {
  if (this == &other)
    return *this;
  subject = other.subject;
  semantics = std::move(other.semantics);
  numeric = other.numeric;
  product = other.product;
  accumulator = other.accumulator;
  coefficients = std::move(other.coefficients);
  consumed = other.consumed;
  other.subject = nullptr;
  other.consumed = true;
  return *this;
}

LogicalResult DistributivePairingPlan::consumeIfValid(Operation *operation,
                                                      ondsp::FixedAttr candidateNumeric,
                                                      ondsp::ProductAttr candidateProduct,
                                                      ondsp::AccType candidateAccumulator,
                                                      ArrayRef<APInt> candidateCoefficients,
                                                      DistributivePairingConsumer consumer) && {
  if (consumed)
    return failure();
  consumed = true;
  if (subject != operation || numeric != candidateNumeric || product != candidateProduct ||
      accumulator != candidateAccumulator ||
      !equalCoefficients(coefficients, candidateCoefficients))
    return failure();
  return consumer(semantics, coefficients);
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

FailureOr<DistributivePairingPlan> FixedPointPrefixRangePlanner::planZeroSeededSymmetricPairing(
    Operation *subject, ondsp::FixedAttr numeric, ondsp::ProductAttr product,
    ondsp::AccType accumulator, ArrayRef<APInt> coefficients) {
  auto numericStorage = dyn_cast<IntegerType>(numeric.getStorage());
  auto accumulatorStorage = dyn_cast<IntegerType>(accumulator.getStorage());
  if (!subject || !numericStorage || !accumulatorStorage || coefficients.size() < 2 ||
      coefficients.size() > static_cast<size_t>(std::numeric_limits<int64_t>::max()))
    return failure();

  SmallVector<FixedPointRawInterval> originalUpdates;
  SmallVector<FixedPointRawInterval> reassociatedUpdates;
  originalUpdates.reserve(coefficients.size());
  reassociatedUpdates.reserve((coefficients.size() + 1) / 2);
  for (const APInt &coefficient : coefficients) {
    FailureOr<FixedPointRawInterval> interval =
        computeSignedFullProductInterval(numeric, coefficient);
    if (failed(interval))
      return failure();
    originalUpdates.push_back(std::move(*interval));
  }

  size_t pairCount = coefficients.size() / 2;
  for (size_t index = 0; index < pairCount; ++index) {
    size_t mirror = coefficients.size() - index - 1;
    if (coefficients[index] != coefficients[mirror])
      return failure();
    FailureOr<FixedPointRawInterval> pairedInterval =
        addFixedPointRawIntervals(originalUpdates[index], originalUpdates[mirror]);
    if (failed(pairedInterval))
      return failure();
    reassociatedUpdates.push_back(std::move(*pairedInterval));
  }
  if (coefficients.size() % 2 != 0)
    reassociatedUpdates.push_back(originalUpdates[pairCount]);

  FixedPointRawInterval initial{APInt(accumulatorStorage.getWidth(), 0),
                                APInt(accumulatorStorage.getWidth(), 0), accumulator.getFrac()};

  FailureOr<ondsp::DistributivePairingSemantics> semantics =
      ondsp::classifyDistributiveProductPairing(subject, numeric, product, accumulator);
  if (failed(semantics) || !semantics->exactBeforeAccumulatorOverflow)
    return failure();

  ondsp::TransformLegality legalityWithoutRangeProof = semantics->legalityWithoutRangeProof;
  if (legalityWithoutRangeProof.isExact()) {
    if (!legalityWithoutRangeProof.isExactWith(ondsp::TransformJustification::FixedWidthModulo))
      return failure();
  } else {
    if (accumulator.getUpdateOverflow() != ondsp::OverflowMode::Saturate ||
        failed(proveAllPrefixesFit(accumulatorStorage.getWidth(), initial, originalUpdates,
                                   reassociatedUpdates)))
      return failure();
  }

  return DistributivePairingPlan(subject, std::move(*semantics), numeric, product, accumulator,
                                 coefficients);
}

} // namespace ondrix::analysis
