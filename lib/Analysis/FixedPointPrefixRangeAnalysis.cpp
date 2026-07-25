#include "ondrix/Analysis/FixedPointPrefixRangeAnalysis.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/MathExtras.h"

#include "mlir/IR/BuiltinTypes.h"

#include <algorithm>
#include <limits>

using namespace mlir;
namespace json = llvm::json;

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

FailureOr<SmallVector<FixedPointRawInterval>>
computePrefixIntervals(unsigned accumulatorWidth, const FixedPointRawInterval &initial,
                       ArrayRef<FixedPointRawInterval> updates) {
  if (accumulatorWidth == 0 || !isValidInterval(initial) ||
      llvm::any_of(
          updates,
          [](const FixedPointRawInterval &interval) { return !isValidInterval(interval); }) ||
      !hasFractionalPosition(updates, initial.frac))
    return failure();
  FailureOr<unsigned> analysisWidth = getAnalysisWidth(accumulatorWidth, initial, updates, {});
  if (failed(analysisWidth))
    return failure();

  SmallVector<FixedPointRawInterval> prefixes;
  prefixes.reserve(updates.size() + 1);
  FixedPointRawInterval prefix = extendInterval(initial, *analysisWidth);
  prefixes.push_back(prefix);
  for (const FixedPointRawInterval &update : updates) {
    FixedPointRawInterval extended = extendInterval(update, *analysisWidth);
    prefix.lower += extended.lower;
    prefix.upper += extended.upper;
    prefixes.push_back(prefix);
  }
  return prefixes;
}

FailureOr<FixedPointRawInterval> computeSignedFullProductIntervalRaw(unsigned storageWidth,
                                                                     unsigned frac,
                                                                     const APInt &coefficient) {
  if (storageWidth == 0 || coefficient.getBitWidth() != storageWidth ||
      storageWidth > std::numeric_limits<unsigned>::max() / 2 ||
      frac > std::numeric_limits<unsigned>::max() / 2)
    return failure();
  unsigned productWidth = storageWidth * 2;
  unsigned productFrac = frac * 2;
  APInt coefficientExtended = coefficient.sext(productWidth);
  APInt first = APInt::getSignedMinValue(storageWidth).sext(productWidth) * coefficientExtended;
  APInt second = APInt::getSignedMaxValue(storageWidth).sext(productWidth) * coefficientExtended;
  if (first.sle(second))
    return FixedPointRawInterval{std::move(first), std::move(second), productFrac};
  return FixedPointRawInterval{std::move(second), std::move(first), productFrac};
}

bool equalIntervals(ArrayRef<FixedPointRawInterval> lhs, ArrayRef<FixedPointRawInterval> rhs) {
  return lhs.size() == rhs.size() &&
         std::equal(lhs.begin(), lhs.end(), rhs.begin(), [](const auto &left, const auto &right) {
           return left.lower == right.lower && left.upper == right.upper && left.frac == right.frac;
         });
}

json::Object apIntToJSON(const APInt &value) {
  SmallString<64> text;
  value.toStringSigned(text);
  return json::Object{{"width", static_cast<int64_t>(value.getBitWidth())},
                      {"value", text.str().str()}};
}

FailureOr<APInt> parseAPInt(const json::Value &value, unsigned maxWidth) {
  const json::Object *object = value.getAsObject();
  if (!object)
    return failure();
  std::optional<int64_t> width = object->getInteger("width");
  std::optional<StringRef> text = object->getString("value");
  if (!width || *width <= 0 || *width > maxWidth || !text || text->empty() ||
      text->size() > static_cast<size_t>(*width) + 1)
    return failure();

  StringRef magnitude = *text;
  bool negative = magnitude.consume_front("-");
  if (magnitude.empty() ||
      llvm::any_of(magnitude, [](char character) { return character < '0' || character > '9'; }))
    return failure();
  APInt parsed(static_cast<unsigned>(*width), magnitude, 10);
  if (negative)
    parsed = -parsed;
  SmallString<64> normalized;
  parsed.toStringSigned(normalized);
  if (normalized != *text)
    return failure();
  return parsed;
}

json::Object intervalToJSON(const FixedPointRawInterval &interval) {
  return json::Object{{"lower", apIntToJSON(interval.lower)},
                      {"upper", apIntToJSON(interval.upper)},
                      {"frac", static_cast<int64_t>(interval.frac)}};
}

FailureOr<FixedPointRawInterval> parseInterval(const json::Value &value, unsigned maxWidth) {
  const json::Object *object = value.getAsObject();
  if (!object)
    return failure();
  const json::Value *lowerValue = object->get("lower");
  const json::Value *upperValue = object->get("upper");
  std::optional<int64_t> frac = object->getInteger("frac");
  if (!lowerValue || !upperValue || !frac || *frac < 0 ||
      *frac > std::numeric_limits<unsigned>::max())
    return failure();
  FailureOr<APInt> lower = parseAPInt(*lowerValue, maxWidth);
  FailureOr<APInt> upper = parseAPInt(*upperValue, maxWidth);
  if (failed(lower) || failed(upper))
    return failure();
  FixedPointRawInterval interval{std::move(*lower), std::move(*upper),
                                 static_cast<unsigned>(*frac)};
  if (!isValidInterval(interval))
    return failure();
  return interval;
}

template <typename Element, typename Parser>
FailureOr<SmallVector<Element>> parseArray(const json::Object &object, StringRef name,
                                           size_t maxElements, Parser parser) {
  const json::Array *array = object.getArray(name);
  if (!array || array->size() > maxElements)
    return failure();
  SmallVector<Element> result;
  result.reserve(array->size());
  for (const json::Value &entry : *array) {
    FailureOr<Element> parsed = parser(entry);
    if (failed(parsed))
      return failure();
    result.push_back(std::move(*parsed));
  }
  return result;
}

} // namespace

FailureOr<FixedPointRawInterval> computeSignedFullProductInterval(ondsp::FixedAttr numeric,
                                                                  const APInt &coefficient) {
  auto storage = dyn_cast<IntegerType>(numeric.getStorage());
  if (!storage || !storage.isSignless() || numeric.getSignedness() != ondsp::Signedness::Signed ||
      coefficient.getBitWidth() != storage.getWidth())
    return failure();
  return computeSignedFullProductIntervalRaw(storage.getWidth(), numeric.getFrac(), coefficient);
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

bool fitsSignedImplementationWidth(const FixedPointRawInterval &interval, unsigned width) {
  if (!isValidInterval(interval) || width == 0)
    return false;
  unsigned comparisonWidth = std::max(width, interval.lower.getBitWidth());
  APInt minimum = APInt::getSignedMinValue(width).sext(comparisonWidth);
  APInt maximum = APInt::getSignedMaxValue(width).sext(comparisonWidth);
  return !interval.lower.sext(comparisonWidth).slt(minimum) &&
         !interval.upper.sext(comparisonWidth).sgt(maximum);
}

json::Object toJSON(const NoOverflowChunkReassociationTrace &trace) {
  json::Array coefficientValues;
  for (const APInt &coefficient : trace.coefficients)
    coefficientValues.emplace_back(apIntToJSON(coefficient));
  json::Array originalPrefixes;
  for (const FixedPointRawInterval &prefix : trace.originalPrefixes)
    originalPrefixes.emplace_back(intervalToJSON(prefix));
  json::Array reassociatedPrefixes;
  for (const FixedPointRawInterval &prefix : trace.reassociatedPrefixes)
    reassociatedPrefixes.emplace_back(intervalToJSON(prefix));

  return json::Object{
      {"schema_version", NoOverflowChunkReassociationTrace::schemaVersion},
      {"kind", "no_overflow_chunk_reassociation"},
      {"subject_ordinal", trace.subjectOrdinal},
      {"numeric_storage_width", static_cast<int64_t>(trace.numericStorageWidth)},
      {"numeric_frac", static_cast<int64_t>(trace.numericFrac)},
      {"numeric_signedness", "signed"},
      {"accumulator_storage_width", static_cast<int64_t>(trace.accumulatorStorageWidth)},
      {"accumulator_frac", static_cast<int64_t>(trace.accumulatorFrac)},
      {"accumulator_signedness", "signed"},
      {"accumulator_update_overflow", "saturate"},
      {"product_selection", "full"},
      {"product_raw_width", static_cast<int64_t>(trace.productRawWidth)},
      {"product_frac", static_cast<int64_t>(trace.productFrac)},
      {"chunk_width", trace.chunkWidth},
      {"coefficients", std::move(coefficientValues)},
      {"original_prefixes", std::move(originalPrefixes)},
      {"reassociated_prefixes", std::move(reassociatedPrefixes)},
  };
}

FailureOr<NoOverflowChunkReassociationTrace>
parseNoOverflowChunkReassociationTrace(const json::Value &value,
                                       NoOverflowChunkReassociationTraceParseLimits limits) {
  const json::Object *object = value.getAsObject();
  if (!object)
    return failure();
  std::optional<int64_t> schema = object->getInteger("schema_version");
  std::optional<StringRef> kind = object->getString("kind");
  std::optional<int64_t> subjectOrdinal = object->getInteger("subject_ordinal");
  std::optional<int64_t> numericStorageWidth = object->getInteger("numeric_storage_width");
  std::optional<int64_t> numericFrac = object->getInteger("numeric_frac");
  std::optional<StringRef> numericSignedness = object->getString("numeric_signedness");
  std::optional<int64_t> accumulatorStorageWidth = object->getInteger("accumulator_storage_width");
  std::optional<int64_t> accumulatorFrac = object->getInteger("accumulator_frac");
  std::optional<StringRef> accumulatorSignedness = object->getString("accumulator_signedness");
  std::optional<StringRef> accumulatorOverflow = object->getString("accumulator_update_overflow");
  std::optional<StringRef> productSelection = object->getString("product_selection");
  std::optional<int64_t> productRawWidth = object->getInteger("product_raw_width");
  std::optional<int64_t> productFrac = object->getInteger("product_frac");
  std::optional<int64_t> chunkWidth = object->getInteger("chunk_width");
  auto isUnsignedField = [](std::optional<int64_t> field) {
    return field && *field >= 0 && *field <= std::numeric_limits<unsigned>::max();
  };
  if (!schema || *schema != NoOverflowChunkReassociationTrace::schemaVersion || !kind ||
      *kind != "no_overflow_chunk_reassociation" || !subjectOrdinal || *subjectOrdinal < 0 ||
      !isUnsignedField(numericStorageWidth) || !isUnsignedField(numericFrac) ||
      !numericSignedness || *numericSignedness != "signed" ||
      !isUnsignedField(accumulatorStorageWidth) || !isUnsignedField(accumulatorFrac) ||
      !accumulatorSignedness || *accumulatorSignedness != "signed" || !accumulatorOverflow ||
      *accumulatorOverflow != "saturate" || !productSelection || *productSelection != "full" ||
      !isUnsignedField(productRawWidth) || !isUnsignedField(productFrac) || !chunkWidth ||
      *chunkWidth <= 1)
    return failure();

  if (limits.maxCoefficients == 0 || limits.maxPrefixes == 0 || limits.maxAPIntWidth == 0)
    return failure();

  FailureOr<SmallVector<APInt>> coefficients = parseArray<APInt>(
      *object, "coefficients", limits.maxCoefficients,
      [&](const json::Value &entry) { return parseAPInt(entry, limits.maxAPIntWidth); });
  FailureOr<SmallVector<FixedPointRawInterval>> originalPrefixes =
      parseArray<FixedPointRawInterval>(
          *object, "original_prefixes", limits.maxPrefixes,
          [&](const json::Value &entry) { return parseInterval(entry, limits.maxAPIntWidth); });
  FailureOr<SmallVector<FixedPointRawInterval>> reassociatedPrefixes =
      parseArray<FixedPointRawInterval>(
          *object, "reassociated_prefixes", limits.maxPrefixes,
          [&](const json::Value &entry) { return parseInterval(entry, limits.maxAPIntWidth); });
  if (failed(coefficients) || failed(originalPrefixes) || failed(reassociatedPrefixes))
    return failure();

  NoOverflowChunkReassociationTrace trace;
  trace.subjectOrdinal = *subjectOrdinal;
  trace.numericStorageWidth = static_cast<unsigned>(*numericStorageWidth);
  trace.numericFrac = static_cast<unsigned>(*numericFrac);
  trace.accumulatorStorageWidth = static_cast<unsigned>(*accumulatorStorageWidth);
  trace.accumulatorFrac = static_cast<unsigned>(*accumulatorFrac);
  trace.productRawWidth = static_cast<unsigned>(*productRawWidth);
  trace.productFrac = static_cast<unsigned>(*productFrac);
  trace.chunkWidth = *chunkWidth;
  trace.coefficients = std::move(*coefficients);
  trace.originalPrefixes = std::move(*originalPrefixes);
  trace.reassociatedPrefixes = std::move(*reassociatedPrefixes);
  return trace;
}

bool areEquivalent(const NoOverflowChunkReassociationTrace &lhs,
                   const NoOverflowChunkReassociationTrace &rhs) {
  return lhs.subjectOrdinal == rhs.subjectOrdinal &&
         lhs.numericStorageWidth == rhs.numericStorageWidth && lhs.numericFrac == rhs.numericFrac &&
         lhs.accumulatorStorageWidth == rhs.accumulatorStorageWidth &&
         lhs.accumulatorFrac == rhs.accumulatorFrac && lhs.productRawWidth == rhs.productRawWidth &&
         lhs.productFrac == rhs.productFrac && lhs.chunkWidth == rhs.chunkWidth &&
         equalCoefficients(lhs.coefficients, rhs.coefficients) &&
         equalIntervals(lhs.originalPrefixes, rhs.originalPrefixes) &&
         equalIntervals(lhs.reassociatedPrefixes, rhs.reassociatedPrefixes);
}

LogicalResult
verifyNoOverflowChunkReassociationTrace(const NoOverflowChunkReassociationTrace &trace) {
  if (trace.subjectOrdinal < 0 || trace.numericStorageWidth == 0 ||
      trace.accumulatorStorageWidth == 0 || trace.chunkWidth <= 1 ||
      trace.numericStorageWidth > std::numeric_limits<unsigned>::max() / 2 ||
      trace.numericFrac > std::numeric_limits<unsigned>::max() / 2 ||
      trace.productRawWidth != trace.numericStorageWidth * 2 ||
      trace.productFrac != trace.numericFrac * 2 || trace.accumulatorFrac != trace.productFrac ||
      trace.coefficients.size() < static_cast<size_t>(trace.chunkWidth))
    return failure();

  SmallVector<FixedPointRawInterval> originalUpdates;
  originalUpdates.reserve(trace.coefficients.size());
  for (const APInt &coefficient : trace.coefficients) {
    FailureOr<FixedPointRawInterval> interval = computeSignedFullProductIntervalRaw(
        trace.numericStorageWidth, trace.numericFrac, coefficient);
    if (failed(interval))
      return failure();
    originalUpdates.push_back(std::move(*interval));
  }

  SmallVector<FixedPointRawInterval> reassociatedUpdates;
  size_t fullChunkCount = trace.coefficients.size() / static_cast<size_t>(trace.chunkWidth);
  reassociatedUpdates.reserve(fullChunkCount +
                              trace.coefficients.size() % static_cast<size_t>(trace.chunkWidth));
  for (size_t chunk = 0; chunk < fullChunkCount; ++chunk) {
    size_t first = chunk * static_cast<size_t>(trace.chunkWidth);
    FixedPointRawInterval partial = originalUpdates[first];
    for (int64_t lane = 1; lane < trace.chunkWidth; ++lane) {
      FailureOr<FixedPointRawInterval> sum =
          addFixedPointRawIntervals(partial, originalUpdates[first + static_cast<size_t>(lane)]);
      if (failed(sum))
        return failure();
      partial = std::move(*sum);
    }
    reassociatedUpdates.push_back(std::move(partial));
  }
  for (size_t index = fullChunkCount * static_cast<size_t>(trace.chunkWidth);
       index < originalUpdates.size(); ++index)
    reassociatedUpdates.push_back(originalUpdates[index]);
  if (llvm::any_of(reassociatedUpdates, [](const FixedPointRawInterval &interval) {
        return !fitsSignedImplementationWidth(interval, 64);
      }))
    return failure();

  FixedPointRawInterval initial{APInt(trace.accumulatorStorageWidth, 0),
                                APInt(trace.accumulatorStorageWidth, 0), trace.accumulatorFrac};
  FailureOr<SmallVector<FixedPointRawInterval>> originalPrefixes =
      computePrefixIntervals(trace.accumulatorStorageWidth, initial, originalUpdates);
  FailureOr<SmallVector<FixedPointRawInterval>> reassociatedPrefixes =
      computePrefixIntervals(trace.accumulatorStorageWidth, initial, reassociatedUpdates);
  return succeeded(originalPrefixes) && succeeded(reassociatedPrefixes) &&
                 equalIntervals(*originalPrefixes, trace.originalPrefixes) &&
                 equalIntervals(*reassociatedPrefixes, trace.reassociatedPrefixes) &&
                 succeeded(FixedPointPrefixRangePlanner::proveAllPrefixesFit(
                     trace.accumulatorStorageWidth, initial, originalUpdates, reassociatedUpdates))
             ? success()
             : failure();
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

NoOverflowChunkReassociationPlan::NoOverflowChunkReassociationPlan(
    NoOverflowChunkReassociationPlan &&other)
    : subject(other.subject), coefficientSource(other.coefficientSource),
      productSemantics(other.productSemantics), numeric(other.numeric), product(other.product),
      accumulator(other.accumulator), coefficients(std::move(other.coefficients)),
      chunkWidth(other.chunkWidth), trace(std::move(other.trace)), consumed(other.consumed) {
  other.subject = nullptr;
  other.coefficientSource = {};
  other.consumed = true;
}

NoOverflowChunkReassociationPlan &
NoOverflowChunkReassociationPlan::operator=(NoOverflowChunkReassociationPlan &&other) {
  if (this == &other)
    return *this;
  subject = other.subject;
  coefficientSource = other.coefficientSource;
  productSemantics = other.productSemantics;
  numeric = other.numeric;
  product = other.product;
  accumulator = other.accumulator;
  coefficients = std::move(other.coefficients);
  chunkWidth = other.chunkWidth;
  trace = std::move(other.trace);
  consumed = other.consumed;
  other.subject = nullptr;
  other.coefficientSource = {};
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

LogicalResult
NoOverflowChunkReassociationPlan::consumeIfValid(ondsp::ReduceMacOp reduction,
                                                 int64_t candidateChunkWidth,
                                                 ChunkReassociationConsumer consumer) && {
  if (consumed)
    return failure();
  consumed = true;
  auto candidateNumeric = dyn_cast<ondsp::FixedAttr>(reduction.getNumeric());
  auto candidateAccumulator = dyn_cast<ondsp::AccType>(reduction.getInitial().getType());
  if (subject != reduction.getOperation() || coefficientSource != reduction.getRhs() ||
      numeric != candidateNumeric || !reduction.getProduct() ||
      product != *reduction.getProduct() || accumulator != candidateAccumulator ||
      !reduction.getInitial().getDefiningOp<ondsp::AccZeroOp>() ||
      chunkWidth != candidateChunkWidth)
    return failure();
  return consumer(productSemantics, coefficients, chunkWidth, trace);
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
  if (semantics->product.rawWidth == std::numeric_limits<unsigned>::max())
    return failure();
  unsigned pairedTermWidth = semantics->product.rawWidth + 1;
  if (llvm::any_of(reassociatedUpdates, [pairedTermWidth](const FixedPointRawInterval &interval) {
        return !fitsSignedImplementationWidth(interval, pairedTermWidth);
      }))
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

FailureOr<NoOverflowChunkReassociationPlan>
FixedPointPrefixRangePlanner::planZeroSeededConstantChunkReduction(
    ondsp::ReduceMacOp reduction, const ondrix::ConstantIntegerMemRefFacts &constant,
    int64_t chunkWidth) {
  auto numeric = dyn_cast<ondsp::FixedAttr>(reduction.getNumeric());
  auto accumulator = dyn_cast<ondsp::AccType>(reduction.getInitial().getType());
  if (!numeric || !accumulator || !reduction.getProduct() ||
      reduction.getRhs() != constant.getSource() ||
      !reduction.getInitial().getDefiningOp<ondsp::AccZeroOp>())
    return failure();
  ondsp::ProductAttr product = *reduction.getProduct();
  ArrayRef<APInt> coefficients = constant.getSequence().getValues();
  auto accumulatorStorage = dyn_cast<IntegerType>(accumulator.getStorage());
  if (!accumulatorStorage || !accumulatorStorage.isSignless() || chunkWidth <= 1 ||
      coefficients.size() < static_cast<size_t>(chunkWidth) ||
      coefficients.size() > static_cast<size_t>(std::numeric_limits<int64_t>::max()) ||
      accumulator.getUpdateOverflow() != ondsp::OverflowMode::Saturate)
    return failure();

  FailureOr<ondsp::ProductSemantics> productSemantics =
      ondsp::inferProductSemantics(reduction, numeric, product);
  if (failed(productSemantics) || productSemantics->selection != ondsp::ProductSelection::Full ||
      accumulator.getSignedness() != numeric.getSignedness() ||
      accumulator.getFrac() != productSemantics->frac)
    return failure();

  SmallVector<FixedPointRawInterval> originalUpdates;
  originalUpdates.reserve(coefficients.size());
  for (const APInt &coefficient : coefficients) {
    FailureOr<FixedPointRawInterval> interval =
        computeSignedFullProductInterval(numeric, coefficient);
    if (failed(interval))
      return failure();
    originalUpdates.push_back(std::move(*interval));
  }

  SmallVector<FixedPointRawInterval> chunkedUpdates;
  size_t fullChunkCount = coefficients.size() / static_cast<size_t>(chunkWidth);
  chunkedUpdates.reserve(fullChunkCount + coefficients.size() % static_cast<size_t>(chunkWidth));
  for (size_t chunk = 0; chunk < fullChunkCount; ++chunk) {
    size_t first = chunk * static_cast<size_t>(chunkWidth);
    FixedPointRawInterval partial = originalUpdates[first];
    for (int64_t lane = 1; lane < chunkWidth; ++lane) {
      FailureOr<FixedPointRawInterval> sum =
          addFixedPointRawIntervals(partial, originalUpdates[first + static_cast<size_t>(lane)]);
      if (failed(sum))
        return failure();
      partial = std::move(*sum);
    }
    chunkedUpdates.push_back(std::move(partial));
  }
  for (size_t index = fullChunkCount * static_cast<size_t>(chunkWidth);
       index < originalUpdates.size(); ++index)
    chunkedUpdates.push_back(originalUpdates[index]);
  if (llvm::any_of(chunkedUpdates, [](const FixedPointRawInterval &interval) {
        return !fitsSignedImplementationWidth(interval, 64);
      }))
    return failure();

  FixedPointRawInterval initial{APInt(accumulatorStorage.getWidth(), 0),
                                APInt(accumulatorStorage.getWidth(), 0), accumulator.getFrac()};
  if (failed(proveAllPrefixesFit(accumulatorStorage.getWidth(), initial, originalUpdates,
                                 chunkedUpdates)))
    return failure();
  FailureOr<SmallVector<FixedPointRawInterval>> originalPrefixes =
      computePrefixIntervals(accumulatorStorage.getWidth(), initial, originalUpdates);
  FailureOr<SmallVector<FixedPointRawInterval>> reassociatedPrefixes =
      computePrefixIntervals(accumulatorStorage.getWidth(), initial, chunkedUpdates);
  if (failed(originalPrefixes) || failed(reassociatedPrefixes))
    return failure();

  auto numericStorage = cast<IntegerType>(numeric.getStorage());
  NoOverflowChunkReassociationTrace trace;
  trace.numericStorageWidth = numericStorage.getWidth();
  trace.numericFrac = numeric.getFrac();
  trace.accumulatorStorageWidth = accumulatorStorage.getWidth();
  trace.accumulatorFrac = accumulator.getFrac();
  trace.productRawWidth = productSemantics->rawWidth;
  trace.productFrac = productSemantics->frac;
  trace.chunkWidth = chunkWidth;
  trace.coefficients.assign(coefficients.begin(), coefficients.end());
  trace.originalPrefixes = std::move(*originalPrefixes);
  trace.reassociatedPrefixes = std::move(*reassociatedPrefixes);
  return NoOverflowChunkReassociationPlan(reduction, constant.getSource(), *productSemantics,
                                          numeric, product, accumulator, coefficients, chunkWidth,
                                          std::move(trace));
}

} // namespace ondrix::analysis
