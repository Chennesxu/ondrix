#include "ondrix/Analysis/FirAlgorithmicWork.h"

#include "llvm/ADT/APInt.h"

using namespace mlir;

namespace ondrix::analysis {

FailureOr<FirAlgorithmicWorkEstimate> estimateOrderedFirWork(int64_t tapCount) {
  if (tapCount < 0)
    return failure();

  FirAlgorithmicWorkEstimate estimate;
  estimate.tapCount = tapCount;
  estimate.scalarInputLoadExecutions = tapCount;
  estimate.scalarCoefficientLoadExecutions = tapCount;
  estimate.scalarProductExecutions = tapCount;
  estimate.accumulatorUpdateExecutions = tapCount;
  return estimate;
}

FirAlgorithmicWorkEstimate
estimateSparseConstantFirWork(const ConstantSequenceFacts &coefficients) {
  FirAlgorithmicWorkEstimate estimate;
  estimate.tapCount = coefficients.getElementCount();
  for (const llvm::APInt &coefficient : coefficients.getValues()) {
    if (coefficient.isZero())
      continue;
    ++estimate.scalarInputLoadExecutions;
    ++estimate.scalarConstantCoefficientUses;
    ++estimate.scalarProductExecutions;
    ++estimate.accumulatorUpdateExecutions;
  }
  return estimate;
}

FailureOr<FirAlgorithmicWorkEstimate>
estimateSymmetricConstantFirWorkAssumingLegal(const ConstantSequenceFacts &coefficients) {
  if (!coefficients.isSymmetric())
    return failure();

  FirAlgorithmicWorkEstimate estimate;
  estimate.tapCount = coefficients.getElementCount();
  llvm::ArrayRef<llvm::APInt> values = coefficients.getValues();
  int64_t pairCount = estimate.tapCount / 2;
  for (int64_t index = 0; index < pairCount; ++index) {
    if (values[index].isZero())
      continue;
    estimate.scalarInputLoadExecutions += 2;
    ++estimate.scalarConstantCoefficientUses;
    ++estimate.scalarProductExecutions;
    ++estimate.widenedPreAddExecutions;
    ++estimate.accumulatorUpdateExecutions;
  }

  if (estimate.tapCount % 2 != 0 && !values[pairCount].isZero()) {
    ++estimate.scalarInputLoadExecutions;
    ++estimate.scalarConstantCoefficientUses;
    ++estimate.scalarProductExecutions;
    ++estimate.accumulatorUpdateExecutions;
  }
  return estimate;
}

FailureOr<FirAlgorithmicWorkEstimate>
estimateVectorFirWorkAssumingUpdateShapeLegal(int64_t tapCount, int64_t vectorWidth,
                                              FirVectorUpdateShape updateShape) {
  if (tapCount < 0 || vectorWidth <= 1)
    return failure();

  FirAlgorithmicWorkEstimate estimate;
  estimate.tapCount = tapCount;
  estimate.vectorWidth = vectorWidth;
  estimate.vectorChunkExecutions = tapCount / vectorWidth;
  estimate.scalarTailElementCount = tapCount % vectorWidth;
  estimate.vectorInputLoadExecutions = estimate.vectorChunkExecutions;
  estimate.vectorCoefficientLoadExecutions = estimate.vectorChunkExecutions;
  estimate.scalarInputLoadExecutions = estimate.scalarTailElementCount;
  estimate.scalarCoefficientLoadExecutions = estimate.scalarTailElementCount;
  estimate.vectorProductExecutions = estimate.vectorChunkExecutions;
  estimate.scalarProductExecutions = estimate.scalarTailElementCount;

  switch (updateShape) {
  case FirVectorUpdateShape::OrderedLanes:
    estimate.accumulatorUpdateExecutions = tapCount;
    break;
  case FirVectorUpdateShape::HorizontalChunks:
    estimate.horizontalReductionExecutions = estimate.vectorChunkExecutions;
    estimate.accumulatorUpdateExecutions =
        estimate.vectorChunkExecutions + estimate.scalarTailElementCount;
    break;
  default:
    return failure();
  }
  return estimate;
}

} // namespace ondrix::analysis
