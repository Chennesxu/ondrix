#include "ondrix/Analysis/ConstantSequenceAnalysis.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/SymbolTable.h"

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

FailureOr<DirectConstantIntegerMemRefFacts>
analyzeDirectConstantIntegerMemRefGlobal(Value value, int64_t maxElements) {
  auto getGlobal = value.getDefiningOp<memref::GetGlobalOp>();
  if (!getGlobal)
    return failure();

  auto global =
      SymbolTable::lookupNearestSymbolFrom<memref::GlobalOp>(getGlobal, getGlobal.getNameAttr());
  if (!global)
    return failure();

  auto initializer = dyn_cast_or_null<DenseIntElementsAttr>(global.getConstantInitValue());
  if (!initializer)
    return failure();
  FailureOr<ConstantSequenceFacts> sequence =
      analyzeConstantIntegerSequence(initializer, maxElements);
  if (failed(sequence))
    return failure();
  return DirectConstantIntegerMemRefFacts(value, std::move(*sequence));
}

} // namespace ondrix
