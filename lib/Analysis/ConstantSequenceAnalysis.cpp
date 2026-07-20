#include "ondrix/Analysis/ConstantSequenceAnalysis.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
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

namespace {

bool isSourceDimension(OpFoldResult size, Value source) {
  auto value = dyn_cast_if_present<Value>(size);
  if (!value)
    return false;
  auto dim = value.getDefiningOp<memref::DimOp>();
  return dim && dim.getSource() == source && dim.getConstantIndex() == 0;
}

bool isFullRangeUnitStrideSubview(memref::SubViewOp subview) {
  auto sourceType = dyn_cast<MemRefType>(subview.getSource().getType());
  auto resultType = dyn_cast<MemRefType>(subview.getResult().getType());
  if (!sourceType || !resultType || sourceType.getRank() != 1 || resultType.getRank() != 1 ||
      sourceType.getElementType() != resultType.getElementType() ||
      sourceType.getMemorySpace() != resultType.getMemorySpace())
    return false;

  ArrayRef<OpFoldResult> offsets = subview.getMixedOffsets();
  ArrayRef<OpFoldResult> sizes = subview.getMixedSizes();
  ArrayRef<OpFoldResult> strides = subview.getMixedStrides();
  if (offsets.size() != 1 || sizes.size() != 1 || strides.size() != 1 ||
      getConstantIntValue(offsets.front()) != 0 || getConstantIntValue(strides.front()) != 1)
    return false;

  if (!sourceType.isDynamicDim(0))
    return getConstantIntValue(sizes.front()) == sourceType.getDimSize(0) ||
           isSourceDimension(sizes.front(), subview.getSource());
  return isSourceDimension(sizes.front(), subview.getSource());
}

bool isMetadataOnlyRankOneCast(memref::CastOp cast) {
  auto sourceType = dyn_cast<MemRefType>(cast.getSource().getType());
  auto resultType = dyn_cast<MemRefType>(cast.getResult().getType());
  return sourceType && resultType && sourceType.getRank() == 1 && resultType.getRank() == 1 &&
         sourceType.getElementType() == resultType.getElementType() &&
         sourceType.getMemorySpace() == resultType.getMemorySpace();
}

FailureOr<Value> resolveConstantGlobalRoot(Value source) {
  Value current = source;
  while (true) {
    if (auto subview = current.getDefiningOp<memref::SubViewOp>()) {
      if (!isFullRangeUnitStrideSubview(subview))
        return failure();
      current = subview.getSource();
      continue;
    }
    if (auto cast = current.getDefiningOp<memref::CastOp>()) {
      if (!isMetadataOnlyRankOneCast(cast))
        return failure();
      current = cast.getSource();
      continue;
    }
    break;
  }
  if (!current.getDefiningOp<memref::GetGlobalOp>())
    return failure();
  return current;
}

} // namespace

FailureOr<ConstantIntegerMemRefFacts> analyzeConstantIntegerMemRef(Value value,
                                                                   int64_t maxElements) {
  FailureOr<Value> root = resolveConstantGlobalRoot(value);
  if (failed(root))
    return failure();
  auto getGlobal = root->getDefiningOp<memref::GetGlobalOp>();
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
  return ConstantIntegerMemRefFacts(value, *root, std::move(*sequence));
}

} // namespace ondrix
