#include "ondrix/Analysis/ReductionWindowAnalysis.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"

using namespace mlir;

namespace ondrix::analysis {

namespace {

/// The loop an operand's window slides with: its induction variable is the
/// subview's own offset, so one trip advances the window by the loop's step.
/// A view offset by anything else does not slide with this loop and is read
/// as stationary rather than guessed at.
std::optional<int64_t> slidingWindowOverlap(Value operand, scf::ForOp loop) {
  auto subview = operand.getDefiningOp<memref::SubViewOp>();
  if (!subview || subview.getMixedOffsets().size() != 1)
    return std::nullopt;

  OpFoldResult offset = subview.getMixedOffsets().front();
  if (dyn_cast_if_present<Value>(offset) != loop.getInductionVar())
    return std::nullopt;

  std::optional<int64_t> length = getConstantIntValue(subview.getMixedSizes().front());
  std::optional<int64_t> stride = getConstantIntValue(subview.getMixedStrides().front());
  std::optional<int64_t> step = getConstantIntValue(loop.getStep());
  if (!length || !stride || !step || *stride != 1 || *step < 1)
    return std::nullopt;
  return *length > *step ? *length - *step : 0;
}

/// A loop bound this analysis may read: a constant, or a dimension of a
/// statically shaped operand, which is what the lowering leaves behind before
/// canonicalization folds it.
std::optional<int64_t> staticIndex(Value value) {
  if (std::optional<int64_t> constant = getConstantIntValue(value))
    return constant;
  auto dim = value.getDefiningOp<tensor::DimOp>();
  auto type = dim ? dyn_cast<RankedTensorType>(dim.getSource().getType()) : RankedTensorType();
  std::optional<int64_t> index = dim ? dim.getConstantIndex() : std::nullopt;
  if (!type || !index || type.isDynamicDim(*index))
    return std::nullopt;
  return type.getDimSize(*index);
}

} // namespace

std::optional<int64_t> getStraightLineCarriedWindow(ondsp::ReduceMacOp reduce) {
  auto loop = reduce->getParentOfType<scf::ForOp>();
  if (!loop)
    return 0;

  int64_t carried = 0;
  for (Value operand : {reduce.getLhs(), reduce.getRhs()}) {
    if (!operand.getDefiningOp<memref::SubViewOp>())
      continue;
    std::optional<int64_t> overlap = slidingWindowOverlap(operand, loop);
    if (!overlap)
      return std::nullopt;
    carried = std::max(carried, *overlap);
  }
  return carried;
}

std::optional<int64_t> getStraightLineCarriedWindow(scf::ForOp accumulatorLoop) {
  auto outer = accumulatorLoop->getParentOfType<scf::ForOp>();
  if (!outer)
    return 0;

  // By this point the window is no longer a view: the lowering folded it into
  // `base[outer + inner]`, the same sliding read expressed as an index, over
  // either a tensor or the buffer it bufferizes to.
  auto readsSlidingIndex = [&](Value index) {
    auto sum = index.getDefiningOp<arith::AddIOp>();
    return sum && ((sum.getLhs() == outer.getInductionVar() &&
                    sum.getRhs() == accumulatorLoop.getInductionVar()) ||
                   (sum.getRhs() == outer.getInductionVar() &&
                    sum.getLhs() == accumulatorLoop.getInductionVar()));
  };
  bool slides =
      accumulatorLoop.getBody()
          ->walk([&](Operation *op) {
            if (auto load = dyn_cast<memref::LoadOp>(op))
              if (load.getIndices().size() == 1 && readsSlidingIndex(load.getIndices().front()))
                return WalkResult::interrupt();
            if (auto extract = dyn_cast<tensor::ExtractOp>(op))
              if (extract.getIndices().size() == 1 &&
                  readsSlidingIndex(extract.getIndices().front()))
                return WalkResult::interrupt();
            return WalkResult::advance();
          })
          .wasInterrupted();
  if (!slides)
    return 0;

  // The window is the tap loop's own trip count, resolved the way the unroll
  // pass resolves its bounds: a dimension of a statically shaped operand is
  // as good as a constant here.
  std::optional<int64_t> lower = staticIndex(accumulatorLoop.getLowerBound());
  std::optional<int64_t> upper = staticIndex(accumulatorLoop.getUpperBound());
  std::optional<int64_t> taps = staticIndex(accumulatorLoop.getStep());
  std::optional<int64_t> advance = staticIndex(outer.getStep());
  if (!lower || !upper || !taps || !advance || *taps != 1 || *advance < 1 || *upper < *lower)
    return std::nullopt;
  int64_t window = *upper - *lower;
  return window > *advance ? window - *advance : 0;
}

} // namespace ondrix::analysis
