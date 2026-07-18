#ifndef ONDRIX_ANALYSIS_FIRALGORITHMICWORK_H
#define ONDRIX_ANALYSIS_FIRALGORITHMICWORK_H

#include "ondrix/Analysis/ConstantSequenceAnalysis.h"

#include "mlir/Support/LLVM.h"

#include <cstdint>

namespace ondrix::analysis {

/// Target-neutral dynamic work executed for one FIR output sample.
///
/// These high-level counts distinguish scalar and Vector work but deliberately
/// omit lower-level extensions, shifts, control flow, instruction costs, and
/// static code size. They are neither transformation-legality evidence nor a
/// complete target cost model.
struct FirAlgorithmicWorkEstimate {
  int64_t tapCount = 0;
  int64_t vectorWidth = 0;
  int64_t vectorChunkExecutions = 0;
  int64_t scalarTailElementCount = 0;

  int64_t scalarInputLoadExecutions = 0;
  int64_t scalarCoefficientLoadExecutions = 0;
  int64_t vectorInputLoadExecutions = 0;
  int64_t vectorCoefficientLoadExecutions = 0;
  int64_t scalarConstantCoefficientUses = 0;

  int64_t scalarProductExecutions = 0;
  int64_t vectorProductExecutions = 0;
  int64_t widenedPreAddExecutions = 0;
  int64_t horizontalReductionExecutions = 0;
  int64_t accumulatorUpdateExecutions = 0;
};

/// Accumulator update shape used only to estimate dynamic Vector work.
enum class FirVectorUpdateShape {
  OrderedLanes,
  HorizontalChunks,
};

/// Estimates dynamic work for the ordinary ordered reduction.
mlir::FailureOr<FirAlgorithmicWorkEstimate> estimateOrderedFirWork(int64_t tapCount);

/// Estimates the current constant zero-tap rewrite after legality has been
/// established independently.
FirAlgorithmicWorkEstimate estimateSparseConstantFirWork(const ConstantSequenceFacts &coefficients);

/// Estimates the current symmetric constant rewrite. Success proves only that
/// the coefficients are symmetric; numeric and range legality must already be
/// established by the fixed-point semantics and range planner.
mlir::FailureOr<FirAlgorithmicWorkEstimate>
estimateSymmetricConstantFirWorkAssumingLegal(const ConstantSequenceFacts &coefficients);

/// Estimates fixed-width Vector chunks followed by a scalar tail. The caller
/// must independently prove that the requested update shape preserves the
/// numeric contract, particularly for `HorizontalChunks`.
mlir::FailureOr<FirAlgorithmicWorkEstimate>
estimateVectorFirWorkAssumingUpdateShapeLegal(int64_t tapCount, int64_t vectorWidth,
                                              FirVectorUpdateShape updateShape);

} // namespace ondrix::analysis

#endif // ONDRIX_ANALYSIS_FIRALGORITHMICWORK_H
