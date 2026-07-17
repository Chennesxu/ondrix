#include "ondrix/Analysis/ConstantSequenceAnalysis.h"

using namespace mlir;

namespace ondrix {

FailureOr<ConstantSequenceFacts> analyzeConstantIntegerSequence(DenseIntElementsAttr elements,
                                                                int64_t maxElements) {
  if (maxElements < 0 || elements.getType().getRank() != 1)
    return failure();

  int64_t elementCount = elements.getNumElements();
  if (elementCount > maxElements)
    return failure();

  ConstantSequenceFacts facts;
  facts.elementCount = elementCount;
  facts.values.reserve(elementCount);
  for (const llvm::APInt &value : elements.getValues<llvm::APInt>()) {
    facts.containsZero |= value.isZero();
    facts.values.push_back(value);
  }

  for (int64_t index = 0; index < elementCount / 2; ++index) {
    if (facts.values[index] != facts.values[elementCount - index - 1]) {
      facts.symmetric = false;
      break;
    }
  }
  return facts;
}

} // namespace ondrix
