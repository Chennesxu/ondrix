#include "ondrix/Conversion/OndspVectorization/OndspVectorization.h"
#include "ondrix/Conversion/Utils/MemRefLayoutUtils.h"

#include "ondrix/Dialect/ondsp/IR/OndspDialect.h"
#include "ondrix/Dialect/ondsp/IR/OndspOps.h"
#include "ondrix/Dialect/ondsp/IR/OndspSemantics.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/SmallVector.h"

#include <optional>

namespace ondrix {
#define GEN_PASS_DEF_VECTORIZEONDSPFPFILTEROUTPUTS
#include "ondrix/Conversion/Passes.h.inc"
} // namespace ondrix

using namespace mlir;

namespace {

/// Largest accepted batch width, bounding every index the rewrite derives.
constexpr int64_t kMaxVectorWidth = 4096;

/// Term-count bound for unrolling a block body; a longer window keeps the
/// rolled loop rather than emitting unbounded straight-line code.
constexpr int64_t kMaxUnrolledTerms = 64;

/// Everything the matcher recovered from one bufferized f32 filter loop.
struct FpFilterLoopShape {
  scf::ForOp loop;
  /// Number of outputs the ordered loop computes.
  int64_t outputLength = 0;
  /// Static coefficient count, which is also the window length.
  int64_t coefficientLength = 0;
  Value input;
  /// The view taps are scalar-loaded from: a contiguous sequence, or a
  /// static reversed subview of one (the convolution coefficient order).
  Value coefficients;
  Value output;
  /// The declared evaluation policy. The exact contracts pin each lane's
  /// event graph verbatim; fast admits both members and never needs R here
  /// because the batch is order-preserving.
  ondrix::ondsp::FpContractMode contract = ondrix::ondsp::FpContractMode::Off;
};

/// Rank-1 memref whose single dimension is contiguous and whose memory space
/// the Vector to LLVM lowering accepts.
bool isBatchableRankOneMemRef(Value value) {
  auto type = dyn_cast<MemRefType>(value.getType());
  return type && type.getRank() == 1 && isLastMemrefDimUnitStride(type) &&
         ondrix::conversion::hasDefaultLLVMVectorMemorySpace(type);
}

/// Statically shaped rank-2 f32 memref, which is all a scalar-loaded matrix
/// has to be.
bool isStaticRankTwoF32MemRef(Value value) {
  auto type = dyn_cast<MemRefType>(value.getType());
  return type && type.getRank() == 2 && !type.isDynamicDim(0) && !type.isDynamicDim(1) &&
         type.getElementType().isF32();
}

/// The same matrix, additionally accessible by a `vector.load` or
/// `vector.store` along its minor dimension.
bool isVectorAccessibleRankTwoMemRef(Value value) {
  if (!isStaticRankTwoF32MemRef(value))
    return false;
  auto type = cast<MemRefType>(value.getType());
  return isLastMemrefDimUnitStride(type) &&
         ondrix::conversion::hasDefaultLLVMVectorMemorySpace(type);
}

/// Skips the layout-erasing casts bufferization inserts between a producer and
/// a dynamic-shaped consumer.
Value lookThroughMemRefCasts(Value value) {
  while (auto cast = value.getDefiningOp<memref::CastOp>())
    value = cast.getSource();
  return value;
}

/// Static length of a rank-1 memref sequence, resolved through casts.
std::optional<int64_t> getStaticRankOneLength(Value value) {
  auto type = dyn_cast<MemRefType>(lookThroughMemRefCasts(value).getType());
  if (!type || type.getRank() != 1 || type.isDynamicDim(0))
    return std::nullopt;
  return type.getDimSize(0);
}

/// Matches the loop shape the valid-boundary f32 filter bufferization emits:
/// one output per iteration, a unit-stride window at offset `m`, one ordered
/// f32 memref reduction from a +0.0 initial value, and one store. Anything
/// else fails closed and keeps the ordered schedule.
FailureOr<FpFilterLoopShape> matchFpFilterLoop(scf::ForOp loop, int64_t vectorWidth) {
  if (!loop.getInitArgs().empty())
    return failure();

  std::optional<int64_t> lowerBound = getConstantIntValue(loop.getLowerBound());
  std::optional<int64_t> upperBound = getConstantIntValue(loop.getUpperBound());
  std::optional<int64_t> step = getConstantIntValue(loop.getStep());
  if (!lowerBound || !upperBound || !step || *lowerBound != 0 || *step != 1 || *upperBound <= 0)
    return failure();

  Block &body = *loop.getBody();
  Value inductionVariable = loop.getInductionVar();

  // Walk the body in order and require exactly the expected operations; an
  // unrecognized operation could carry a side effect or a second use the
  // batched schedule would reorder.
  SmallVector<Operation *> operations;
  for (Operation &operation : body.without_terminator()) {
    if (isa<memref::CastOp>(operation))
      continue;
    operations.push_back(&operation);
  }
  if (operations.size() != 3)
    return failure();

  auto window = dyn_cast<memref::SubViewOp>(operations[0]);
  auto reduce = dyn_cast<ondrix::ondsp::ReduceMacOp>(operations[1]);
  auto store = dyn_cast<memref::StoreOp>(operations[2]);
  if (!window || !reduce || !store)
    return failure();

  if (store.getValueToStore() != reduce.getResult() || store.getIndices().size() != 1 ||
      store.getIndices().front() != inductionVariable)
    return failure();
  if (!reduce.getResult().hasOneUse())
    return failure();
  if (!window.getResult().hasOneUse() || !reduce.getLhs().hasOneUse())
    return failure();
  if (lookThroughMemRefCasts(reduce.getLhs()) != window.getResult())
    return failure();

  // The exact contracts pin each lane's event graph, which the batched form
  // preserves verbatim. A fast site is admitted on the same structural
  // ground: order preservation is inside every declared set.
  auto numeric = dyn_cast<ondrix::ondsp::FpAttr>(reduce.getNumeric());
  if (!numeric || !numeric.getFormat().isF32())
    return failure();

  // The ordered reduction starts from +0.0 and the batched lanes must start
  // from the same value. The sign matters: -0.0 is a different f32 value and
  // a different first-addition result under off.
  FloatAttr initial;
  if (!matchPattern(reduce.getInitial(), m_Constant(&initial)) || !initial.getType().isF32() ||
      !initial.getValue().isPosZero())
    return failure();

  // The window must be a plain unit-stride rank-1 slice at offset `m`.
  if (window.getType().getRank() != 1 || window.getMixedOffsets().size() != 1 ||
      window.getMixedSizes().size() != 1 || window.getMixedStrides().size() != 1)
    return failure();
  auto windowOffset = window.getMixedOffsets().front().dyn_cast<Value>();
  if (!windowOffset || windowOffset != inductionVariable)
    return failure();
  std::optional<int64_t> windowStride = getConstantIntValue(window.getMixedStrides().front());
  std::optional<int64_t> windowLength = getConstantIntValue(window.getMixedSizes().front());
  if (!windowStride || *windowStride != 1 || !windowLength || *windowLength <= 0)
    return failure();

  FpFilterLoopShape shape;
  shape.loop = loop;
  shape.outputLength = *upperBound;
  shape.coefficientLength = *windowLength;
  shape.input = window.getSource();
  shape.coefficients = reduce.getRhs();
  shape.output = store.getMemRef();
  shape.contract = numeric.getContract();

  std::optional<int64_t> coefficientLength = getStaticRankOneLength(shape.coefficients);
  if (!coefficientLength || *coefficientLength != shape.coefficientLength)
    return failure();
  shape.coefficients = lookThroughMemRefCasts(shape.coefficients);

  // Taps are scalar-loaded, so the coefficient view only has to be a static
  // rank-1 sequence: contiguous, or a whole-length static subview at stride
  // +-1 of one (a reversed view is the convolution coefficient order; the
  // touched index range is pinned below, not assumed).
  if (!isBatchableRankOneMemRef(shape.coefficients)) {
    auto view = shape.coefficients.getDefiningOp<memref::SubViewOp>();
    if (!view || view.getType().getRank() != 1 ||
        !ondrix::conversion::hasDefaultLLVMVectorMemorySpace(view.getType()) ||
        !isBatchableRankOneMemRef(view.getSource()))
      return failure();
    std::optional<int64_t> offset = getConstantIntValue(view.getMixedOffsets().front());
    std::optional<int64_t> stride = getConstantIntValue(view.getMixedStrides().front());
    std::optional<int64_t> size = getConstantIntValue(view.getMixedSizes().front());
    std::optional<int64_t> sourceLength = getStaticRankOneLength(view.getSource());
    if (!offset || !stride || !size || !sourceLength || *size != shape.coefficientLength ||
        (*stride != 1 && *stride != -1))
      return failure();
    int64_t first = *offset;
    int64_t last = *offset + *stride * (*size - 1);
    if (std::min(first, last) < 0 || std::max(first, last) >= *sourceLength)
      return failure();
  }

  if (!isBatchableRankOneMemRef(shape.input) || !isBatchableRankOneMemRef(shape.output))
    return failure();
  if (shape.coefficients.getParentBlock() == &body || shape.input.getParentBlock() == &body ||
      shape.output.getParentBlock() == &body)
    return failure();
  if (cast<MemRefType>(shape.input.getType()).getElementType() !=
          Float32Type::get(loop.getContext()) ||
      cast<MemRefType>(shape.coefficients.getType()).getElementType() !=
          Float32Type::get(loop.getContext()) ||
      cast<MemRefType>(shape.output.getType()).getElementType() !=
          Float32Type::get(loop.getContext()))
    return failure();

  // The rewrite moves a block's W stores past all of that block's tap loads,
  // so it is only sound when the three sequences are distinct storage — the
  // same obligation, refusal set, and entry-argument ABI residual as the
  // fixed-point decimate batching, discharged by the same analysis.
  if (ondrix::conversion::mayShareStorage(shape.input, shape.output) ||
      ondrix::conversion::mayShareStorage(shape.coefficients, shape.output) ||
      ondrix::conversion::mayShareStorage(shape.input, shape.coefficients))
    return failure();

  // No widened span, unlike the decimate batching (window-union argument in
  // the pass description).
  auto inputType = cast<MemRefType>(shape.input.getType());
  if (inputType.isDynamicDim(0))
    return failure();
  int64_t inputLength = inputType.getDimSize(0);
  if (inputLength <= 0 || shape.outputLength > inputLength || shape.coefficientLength > inputLength)
    return failure();
  if (inputLength < shape.outputLength + shape.coefficientLength - 1)
    return failure();
  int64_t fullBlocks = shape.outputLength / vectorWidth;
  if (fullBlocks < 1)
    return failure();

  // Independently of the argument above, pin the actual extent of the last
  // batched load.
  int64_t lastLoadEnd =
      (fullBlocks - 1) * vectorWidth + shape.coefficientLength - 1 + vectorWidth - 1;
  if (lastLoadEnd >= inputLength)
    return failure();

  return shape;
}

/// Replaces the leading full blocks of an ordered f32 filter loop with a
/// batched loop over `vectorWidth` outputs at a time and moves the ordered
/// loop's lower bound past them. Each lane runs the declared per-output event
/// graph unchanged — tap order, one product and one accumulation event per
/// tap, +0.0 initial value — so the authorization is structural (per-lane
/// event-graph identity), not algebraic, and holds for both exact contracts.
/// One per-term lane update in the caller's declared operand order; the fast
/// member selection and its F spend live here so every batcher agrees.
Value createLaneUpdate(OpBuilder &builder, Location loc, Value lhs, Value rhs, Value accumulator,
                       ondrix::ondsp::FpContractMode contract, bool fuseFast) {
  if (contract == ondrix::ondsp::FpContractMode::Fma)
    return builder.create<math::FmaOp>(loc, lhs, rhs, accumulator);
  if (contract == ondrix::ondsp::FpContractMode::Fast && fuseFast) {
    // fast admits both members; selecting the fused one spends F, and the
    // order-preserving batch never spends R.
    return ondrix::ondsp::consumeFastPermission(
        builder.create<math::FmaOp>(loc, lhs, rhs, accumulator),
        ondrix::ondsp::FastPermission::FuseMultiplyAdd);
  }
  Value product = builder.create<arith::MulFOp>(loc, lhs, rhs);
  return builder.create<arith::AddFOp>(loc, accumulator, product);
}

/// blockStart advanced by a compile-time term offset.
Value createTermBase(OpBuilder &builder, Location loc, Value blockStart, int64_t term) {
  if (term == 0)
    return blockStart;
  Value offset = builder.create<arith::ConstantIndexOp>(loc, term);
  return builder.create<arith::AddIOp>(loc, blockStart, offset);
}

void batchFpFilterOutputs(const FpFilterLoopShape &shape, int64_t vectorWidth, bool fuseFast,
                          OpBuilder &builder) {
  scf::ForOp loop = shape.loop;
  Location loc = loop.getLoc();
  int64_t fullBlocks = shape.outputLength / vectorWidth;
  int64_t batchedOutputs = fullBlocks * vectorWidth;

  auto laneType = VectorType::get({vectorWidth}, builder.getF32Type());

  builder.setInsertionPoint(loop);
  Value zeroIndex = builder.create<arith::ConstantIndexOp>(loc, 0);
  Value oneIndex = builder.create<arith::ConstantIndexOp>(loc, 1);
  Value tapEnd = builder.create<arith::ConstantIndexOp>(loc, shape.coefficientLength);
  Value batchedEnd = builder.create<arith::ConstantIndexOp>(loc, batchedOutputs);
  Value batchStep = builder.create<arith::ConstantIndexOp>(loc, vectorWidth);
  Value laneZero = builder.create<arith::ConstantOp>(
      loc, laneType, DenseElementsAttr::get(laneType, builder.getF32FloatAttr(0.0f)));

  // The coefficients are block-invariant reads of storage the matcher proved
  // distinct from the output, so their splats hoist above the batched loop
  // and the taps unroll: one lane update per tap, no per-tap reload.
  bool unrolled = shape.coefficientLength <= kMaxUnrolledTerms;
  SmallVector<Value> tapSplats;
  if (unrolled) {
    for (int64_t tap = 0; tap < shape.coefficientLength; ++tap) {
      Value index = builder.create<arith::ConstantIndexOp>(loc, tap);
      Value coefficient =
          builder.create<memref::LoadOp>(loc, shape.coefficients, ValueRange{index});
      tapSplats.push_back(builder.create<vector::SplatOp>(loc, laneType, coefficient));
    }
  }

  builder.create<scf::ForOp>(
      loc, zeroIndex, batchedEnd, batchStep, ValueRange{},
      [&](OpBuilder &blockBuilder, Location blockLoc, Value blockStart, ValueRange) {
        Value lanes;
        if (unrolled) {
          lanes = laneZero;
          for (int64_t tap = 0; tap < shape.coefficientLength; ++tap) {
            Value base = createTermBase(blockBuilder, blockLoc, blockStart, tap);
            Value values = blockBuilder.create<vector::LoadOp>(blockLoc, laneType, shape.input,
                                                               ValueRange{base});
            lanes = createLaneUpdate(blockBuilder, blockLoc, values, tapSplats[tap], lanes,
                                     shape.contract, fuseFast);
          }
        } else {
          auto taps = blockBuilder.create<scf::ForOp>(
              blockLoc, zeroIndex, tapEnd, oneIndex, ValueRange{laneZero},
              [&](OpBuilder &tapBuilder, Location tapLoc, Value tap, ValueRange iterArgs) {
                Value base = tapBuilder.create<arith::AddIOp>(tapLoc, blockStart, tap);
                Value values = tapBuilder.create<vector::LoadOp>(tapLoc, laneType, shape.input,
                                                                 ValueRange{base});
                Value coefficient =
                    tapBuilder.create<memref::LoadOp>(tapLoc, shape.coefficients, ValueRange{tap});
                Value splat = tapBuilder.create<vector::SplatOp>(tapLoc, laneType, coefficient);
                tapBuilder.create<scf::YieldOp>(tapLoc, createLaneUpdate(tapBuilder, tapLoc, values,
                                                                         splat, iterArgs.front(),
                                                                         shape.contract, fuseFast));
              });
          lanes = taps.getResult(0);
        }
        blockBuilder.create<vector::StoreOp>(blockLoc, lanes, shape.output, ValueRange{blockStart});
        blockBuilder.create<scf::YieldOp>(blockLoc);
      });

  // The ordered loop keeps its body and now starts at the first output the
  // batched loop did not produce. A fully covered loop is erased instead:
  // a dead residual body would still lower and record a spend the audit
  // can never observe.
  if (batchedOutputs == shape.outputLength) {
    loop.erase();
    return;
  }
  loop.getLowerBoundMutable().assign(batchedEnd);
}

/// Everything the matcher recovered from one bufferized windowed-sum loop
/// (the f32 moving-average shape: seed load, ordered adds, one division).
struct FpWindowSumLoopShape {
  scf::ForOp loop;
  int64_t outputLength = 0;
  int64_t windowLength = 0;
  Value input;
  Value output;
  Value divisor;
};

/// Matches the loop the f32 moving-average bufferization emits: per output,
/// a seed load at the output index, a left-to-right add loop over the window
/// tail, one division by a loop-invariant constant, and one store. Anything
/// else fails closed and keeps the ordered schedule.
FailureOr<FpWindowSumLoopShape> matchFpWindowSumLoop(scf::ForOp loop, int64_t vectorWidth) {
  if (!loop.getInitArgs().empty())
    return failure();

  std::optional<int64_t> lowerBound = getConstantIntValue(loop.getLowerBound());
  std::optional<int64_t> upperBound = getConstantIntValue(loop.getUpperBound());
  std::optional<int64_t> step = getConstantIntValue(loop.getStep());
  if (!lowerBound || !upperBound || !step || *lowerBound != 0 || *step != 1 || *upperBound <= 0)
    return failure();

  Block &body = *loop.getBody();
  Value outputIndex = loop.getInductionVar();

  SmallVector<Operation *> operations;
  for (Operation &operation : body.without_terminator())
    operations.push_back(&operation);
  if (operations.size() != 4)
    return failure();

  auto seed = dyn_cast<memref::LoadOp>(operations[0]);
  auto window = dyn_cast<scf::ForOp>(operations[1]);
  auto mean = dyn_cast<arith::DivFOp>(operations[2]);
  auto store = dyn_cast<memref::StoreOp>(operations[3]);
  if (!seed || !window || !mean || !store)
    return failure();

  if (seed.getIndices().size() != 1 || seed.getIndices().front() != outputIndex ||
      !seed.getResult().hasOneUse())
    return failure();
  if (window.getInitArgs().size() != 1 || window.getInitArgs().front() != seed.getResult() ||
      !window.getResult(0).hasOneUse())
    return failure();
  std::optional<int64_t> windowLower = getConstantIntValue(window.getLowerBound());
  std::optional<int64_t> windowUpper = getConstantIntValue(window.getUpperBound());
  std::optional<int64_t> windowStep = getConstantIntValue(window.getStep());
  if (!windowLower || !windowUpper || !windowStep || *windowLower != 1 || *windowStep != 1 ||
      *windowUpper <= 1)
    return failure();

  // The window tail: position = outputIndex + term, one load, one ordered add.
  Block &windowBody = *window.getBody();
  SmallVector<Operation *> tail;
  for (Operation &operation : windowBody.without_terminator())
    tail.push_back(&operation);
  if (tail.size() != 3)
    return failure();
  auto position = dyn_cast<arith::AddIOp>(tail[0]);
  auto load = dyn_cast<memref::LoadOp>(tail[1]);
  auto add = dyn_cast<arith::AddFOp>(tail[2]);
  if (!position || !load || !add)
    return failure();
  if (position.getLhs() != outputIndex || position.getRhs() != window.getInductionVar())
    return failure();
  if (load.getIndices().size() != 1 || load.getIndices().front() != position.getResult() ||
      load.getMemRef() != seed.getMemRef())
    return failure();
  if (add.getLhs() != windowBody.getArgument(1) || add.getRhs() != load.getResult())
    return failure();
  auto yield = cast<scf::YieldOp>(windowBody.getTerminator());
  if (yield.getOperand(0) != add.getResult())
    return failure();

  FloatAttr divisor;
  if (mean.getLhs() != window.getResult(0) || !matchPattern(mean.getRhs(), m_Constant(&divisor)) ||
      !divisor.getType().isF32())
    return failure();
  if (store.getValueToStore() != mean.getResult() || store.getIndices().size() != 1 ||
      store.getIndices().front() != outputIndex)
    return failure();

  FpWindowSumLoopShape shape;
  shape.loop = loop;
  shape.outputLength = *upperBound;
  shape.windowLength = *windowUpper;
  shape.input = seed.getMemRef();
  shape.output = store.getMemRef();
  shape.divisor = mean.getRhs();

  if (!isBatchableRankOneMemRef(shape.input) || !isBatchableRankOneMemRef(shape.output))
    return failure();
  if (shape.input.getParentBlock() == &body || shape.output.getParentBlock() == &body ||
      shape.divisor.getParentBlock() == &body)
    return failure();
  if (cast<MemRefType>(shape.input.getType()).getElementType() !=
          Float32Type::get(loop.getContext()) ||
      cast<MemRefType>(shape.output.getType()).getElementType() !=
          Float32Type::get(loop.getContext()))
    return failure();

  // Same deferred-store obligation and window-union extent argument as the
  // filter batching: block m reads exactly the union of its W windows.
  if (ondrix::conversion::mayShareStorage(shape.input, shape.output))
    return failure();
  auto inputType = cast<MemRefType>(shape.input.getType());
  if (inputType.isDynamicDim(0))
    return failure();
  int64_t inputLength = inputType.getDimSize(0);
  if (inputLength < shape.outputLength + shape.windowLength - 1)
    return failure();
  int64_t fullBlocks = shape.outputLength / vectorWidth;
  if (fullBlocks < 1)
    return failure();
  int64_t lastLoadEnd = (fullBlocks - 1) * vectorWidth + shape.windowLength - 1 + vectorWidth - 1;
  if (lastLoadEnd >= inputLength)
    return failure();

  return shape;
}

/// Batches W windowed-sum outputs per vector: each lane runs its declared
/// per-output events verbatim (seed element, left-to-right adds, one
/// division), so the authorization is the same per-lane event-graph identity
/// as the filter batching, for every contract.
void batchFpWindowSumOutputs(const FpWindowSumLoopShape &shape, int64_t vectorWidth,
                             OpBuilder &builder) {
  scf::ForOp loop = shape.loop;
  Location loc = loop.getLoc();
  int64_t fullBlocks = shape.outputLength / vectorWidth;
  int64_t batchedOutputs = fullBlocks * vectorWidth;

  auto laneType = VectorType::get({vectorWidth}, builder.getF32Type());

  builder.setInsertionPoint(loop);
  Value zeroIndex = builder.create<arith::ConstantIndexOp>(loc, 0);
  Value oneIndex = builder.create<arith::ConstantIndexOp>(loc, 1);
  Value windowEnd = builder.create<arith::ConstantIndexOp>(loc, shape.windowLength);
  Value batchedEnd = builder.create<arith::ConstantIndexOp>(loc, batchedOutputs);
  Value batchStep = builder.create<arith::ConstantIndexOp>(loc, vectorWidth);
  Value divisorLanes = builder.create<vector::SplatOp>(loc, laneType, shape.divisor);

  bool unrolled = shape.windowLength <= kMaxUnrolledTerms;
  builder.create<scf::ForOp>(
      loc, zeroIndex, batchedEnd, batchStep, ValueRange{},
      [&](OpBuilder &blockBuilder, Location blockLoc, Value blockStart, ValueRange) {
        Value sums =
            blockBuilder.create<vector::LoadOp>(blockLoc, laneType, shape.input, blockStart);
        if (unrolled) {
          for (int64_t term = 1; term < shape.windowLength; ++term) {
            Value base = createTermBase(blockBuilder, blockLoc, blockStart, term);
            Value values =
                blockBuilder.create<vector::LoadOp>(blockLoc, laneType, shape.input, base);
            sums = blockBuilder.create<arith::AddFOp>(blockLoc, sums, values);
          }
        } else {
          auto terms = blockBuilder.create<scf::ForOp>(
              blockLoc, oneIndex, windowEnd, oneIndex, ValueRange{sums},
              [&](OpBuilder &termBuilder, Location termLoc, Value term, ValueRange iterArgs) {
                Value base = termBuilder.create<arith::AddIOp>(termLoc, blockStart, term);
                Value values =
                    termBuilder.create<vector::LoadOp>(termLoc, laneType, shape.input, base);
                termBuilder.create<scf::YieldOp>(
                    termLoc, termBuilder.create<arith::AddFOp>(termLoc, iterArgs.front(), values)
                                 .getResult());
              });
          sums = terms.getResult(0);
        }
        Value means = blockBuilder.create<arith::DivFOp>(blockLoc, sums, divisorLanes);
        blockBuilder.create<vector::StoreOp>(blockLoc, means, shape.output, blockStart);
        blockBuilder.create<scf::YieldOp>(blockLoc);
      });

  if (batchedOutputs == shape.outputLength) {
    loop.erase();
    return;
  }
  loop.getLowerBoundMutable().assign(batchedEnd);
}

/// Everything the matcher recovered from one bufferized f32 matmul column
/// loop: the accumulator loop over the inner axis plus its one store.
struct FpColumnTileLoopShape {
  scf::ForOp loop;
  int64_t columnCount = 0;
  int64_t innerCount = 0;
  Value lhs;
  Value rhs;
  Value output;
  /// The column-invariant row index, available outside the column loop.
  Value rowIndex;
  /// Recovered from the emitted events rather than from an attribute: there is
  /// no contract-carrying operation left after bufferization. A fused event
  /// already carrying a spend record is a fast site.
  ondrix::ondsp::FpContractMode contract = ondrix::ondsp::FpContractMode::Off;
};

/// Whether `value` is available where the loop is, rather than produced inside
/// it.
bool isAvailableAtLoop(scf::ForOp loop, Value value) {
  return !loop.getRegion().isAncestor(value.getParentRegion());
}

/// Matches the loop shape the f32 matmul bufferization emits for one output
/// row: per column, an ordered f32 accumulator loop over the inner axis from
/// a +0.0 initial value with a column-invariant scalar `A[i,k]` and a
/// unit-stride `B[k,j]`, and one store. Anything else fails closed and keeps
/// the ordered schedule.
FailureOr<FpColumnTileLoopShape> matchFpColumnTileLoop(scf::ForOp loop, int64_t vectorWidth) {
  if (!loop.getInitArgs().empty())
    return failure();

  std::optional<int64_t> lowerBound = getConstantIntValue(loop.getLowerBound());
  std::optional<int64_t> upperBound = getConstantIntValue(loop.getUpperBound());
  std::optional<int64_t> step = getConstantIntValue(loop.getStep());
  if (!lowerBound || !upperBound || !step || *lowerBound != 0 || *step != 1 || *upperBound <= 0)
    return failure();

  Block &body = *loop.getBody();
  Value columnIndex = loop.getInductionVar();

  // The accumulator seed is a constant that may or may not have been hoisted
  // out of this body yet; nothing else may stand beside the accumulator loop
  // and its store.
  SmallVector<Operation *> operations;
  for (Operation &operation : body.without_terminator()) {
    if (isa<arith::ConstantOp>(operation))
      continue;
    operations.push_back(&operation);
  }
  if (operations.size() != 2)
    return failure();

  auto terms = dyn_cast<scf::ForOp>(operations[0]);
  auto store = dyn_cast<memref::StoreOp>(operations[1]);
  if (!terms || !store)
    return failure();

  if (terms.getInitArgs().size() != 1 || !terms.getResult(0).hasOneUse())
    return failure();
  std::optional<int64_t> innerLower = getConstantIntValue(terms.getLowerBound());
  std::optional<int64_t> innerUpper = getConstantIntValue(terms.getUpperBound());
  std::optional<int64_t> innerStep = getConstantIntValue(terms.getStep());
  if (!innerLower || !innerUpper || !innerStep || *innerLower != 0 || *innerStep != 1 ||
      *innerUpper <= 0)
    return failure();

  // The batched lanes must start from the same value: -0.0 is a different f32
  // value and a different first-addition result under off.
  FloatAttr initial;
  if (!matchPattern(terms.getInitArgs().front(), m_Constant(&initial)) ||
      !initial.getType().isF32() || !initial.getValue().isPosZero())
    return failure();

  if (store.getValueToStore() != terms.getResult(0) || store.getIndices().size() != 2 ||
      store.getIndices()[1] != columnIndex)
    return failure();
  Value rowIndex = store.getIndices()[0];

  Block &termBody = *terms.getBody();
  if (termBody.getNumArguments() != 2)
    return failure();
  SmallVector<Operation *> events;
  for (Operation &operation : termBody.without_terminator())
    events.push_back(&operation);
  if (events.size() != 3 && events.size() != 4)
    return failure();

  auto left = dyn_cast<memref::LoadOp>(events[0]);
  auto right = dyn_cast<memref::LoadOp>(events[1]);
  if (!left || !right || !left.getResult().hasOneUse() || !right.getResult().hasOneUse())
    return failure();
  if (left.getIndices().size() != 2 || left.getIndices()[0] != rowIndex ||
      left.getIndices()[1] != terms.getInductionVar())
    return failure();
  if (right.getIndices().size() != 2 || right.getIndices()[0] != terms.getInductionVar() ||
      right.getIndices()[1] != columnIndex)
    return failure();

  Value accumulator = termBody.getArgument(1);
  auto yield = cast<scf::YieldOp>(termBody.getTerminator());
  ondrix::ondsp::FpContractMode contract;
  if (events.size() == 4) {
    auto product = dyn_cast<arith::MulFOp>(events[2]);
    auto sum = dyn_cast<arith::AddFOp>(events[3]);
    if (!product || !sum || product.getLhs() != left.getResult() ||
        product.getRhs() != right.getResult() || !product.getResult().hasOneUse() ||
        sum.getLhs() != accumulator || sum.getRhs() != product.getResult() ||
        yield.getOperand(0) != sum.getResult())
      return failure();
    contract = ondrix::ondsp::FpContractMode::Off;
  } else {
    auto fused = dyn_cast<math::FmaOp>(events[2]);
    if (!fused || fused.getA() != left.getResult() || fused.getB() != right.getResult() ||
        fused.getC() != accumulator || yield.getOperand(0) != fused.getResult())
      return failure();
    contract =
        ondrix::ondsp::hasSpentFastPermission(fused, ondrix::ondsp::FastPermission::FuseMultiplyAdd)
            ? ondrix::ondsp::FpContractMode::Fast
            : ondrix::ondsp::FpContractMode::Fma;
  }

  FpColumnTileLoopShape shape;
  shape.loop = loop;
  shape.columnCount = *upperBound;
  shape.innerCount = *innerUpper;
  shape.lhs = left.getMemRef();
  shape.rhs = right.getMemRef();
  shape.output = store.getMemRef();
  shape.rowIndex = rowIndex;
  shape.contract = contract;

  if (!isStaticRankTwoF32MemRef(shape.lhs) || !isVectorAccessibleRankTwoMemRef(shape.rhs) ||
      !isVectorAccessibleRankTwoMemRef(shape.output))
    return failure();
  for (Value value : {shape.lhs, shape.rhs, shape.output, shape.rowIndex})
    if (!isAvailableAtLoop(loop, value))
      return failure();

  // The rewrite moves a block's W stores past all of that block's loads, so it
  // needs the same statically distinct storage, refusal set, and entry-argument
  // ABI residual as the filter batching. The two read sequences may alias.
  if (ondrix::conversion::mayShareStorage(shape.lhs, shape.output) ||
      ondrix::conversion::mayShareStorage(shape.rhs, shape.output))
    return failure();

  int64_t fullBlocks = shape.columnCount / vectorWidth;
  if (fullBlocks < 1)
    return failure();

  // Pin the actual extent of the last batched access rather than inferring it
  // from the loop bound.
  int64_t lastColumn = fullBlocks * vectorWidth - 1;
  if (lastColumn >= cast<MemRefType>(shape.rhs.getType()).getDimSize(1) ||
      lastColumn >= cast<MemRefType>(shape.output.getType()).getDimSize(1) ||
      shape.innerCount > cast<MemRefType>(shape.lhs.getType()).getDimSize(1) ||
      shape.innerCount > cast<MemRefType>(shape.rhs.getType()).getDimSize(0))
    return failure();

  return shape;
}

/// Batches W columns of one matmul output row into vector lanes: the inner
/// axis stays a loop carrying a vector accumulator, `A[i,k]` is broadcast, and
/// `B[k,j]` becomes one contiguous load. Each lane runs its declared
/// per-output events verbatim — +0.0 initial value, ascending inner index, one
/// update event per term — which is the same per-lane event-graph identity the
/// filter batching relies on.
void batchFpColumnTiles(const FpColumnTileLoopShape &shape, int64_t vectorWidth, bool fuseFast,
                        OpBuilder &builder) {
  scf::ForOp loop = shape.loop;
  Location loc = loop.getLoc();
  int64_t fullBlocks = shape.columnCount / vectorWidth;
  int64_t batchedColumns = fullBlocks * vectorWidth;

  auto laneType = VectorType::get({vectorWidth}, builder.getF32Type());

  builder.setInsertionPoint(loop);
  Value zeroIndex = builder.create<arith::ConstantIndexOp>(loc, 0);
  Value oneIndex = builder.create<arith::ConstantIndexOp>(loc, 1);
  Value innerEnd = builder.create<arith::ConstantIndexOp>(loc, shape.innerCount);
  Value batchedEnd = builder.create<arith::ConstantIndexOp>(loc, batchedColumns);
  Value batchStep = builder.create<arith::ConstantIndexOp>(loc, vectorWidth);
  Value laneZero = builder.create<arith::ConstantOp>(
      loc, laneType, DenseElementsAttr::get(laneType, builder.getF32FloatAttr(0.0f)));

  // The left operand's row is column-block-invariant, so its splats hoist
  // above the batched loop and the inner axis unrolls, mirroring the filter
  // batcher's tap treatment.
  bool unrolled = shape.innerCount <= kMaxUnrolledTerms;
  SmallVector<Value> rowSplats;
  if (unrolled) {
    for (int64_t term = 0; term < shape.innerCount; ++term) {
      Value index = builder.create<arith::ConstantIndexOp>(loc, term);
      Value element =
          builder.create<memref::LoadOp>(loc, shape.lhs, ValueRange{shape.rowIndex, index});
      rowSplats.push_back(builder.create<vector::SplatOp>(loc, laneType, element));
    }
  }

  builder.create<scf::ForOp>(
      loc, zeroIndex, batchedEnd, batchStep, ValueRange{},
      [&](OpBuilder &blockBuilder, Location blockLoc, Value blockStart, ValueRange) {
        Value lanes;
        if (unrolled) {
          lanes = laneZero;
          for (int64_t term = 0; term < shape.innerCount; ++term) {
            Value index = blockBuilder.create<arith::ConstantIndexOp>(blockLoc, term);
            Value values = blockBuilder.create<vector::LoadOp>(blockLoc, laneType, shape.rhs,
                                                               ValueRange{index, blockStart});
            lanes = createLaneUpdate(blockBuilder, blockLoc, rowSplats[term], values, lanes,
                                     shape.contract, fuseFast);
          }
        } else {
          auto terms = blockBuilder.create<scf::ForOp>(
              blockLoc, zeroIndex, innerEnd, oneIndex, ValueRange{laneZero},
              [&](OpBuilder &termBuilder, Location termLoc, Value index, ValueRange iterArgs) {
                Value element = termBuilder.create<memref::LoadOp>(
                    termLoc, shape.lhs, ValueRange{shape.rowIndex, index});
                Value splat = termBuilder.create<vector::SplatOp>(termLoc, laneType, element);
                Value values = termBuilder.create<vector::LoadOp>(termLoc, laneType, shape.rhs,
                                                                  ValueRange{index, blockStart});
                termBuilder.create<scf::YieldOp>(
                    termLoc, createLaneUpdate(termBuilder, termLoc, splat, values, iterArgs.front(),
                                              shape.contract, fuseFast));
              });
          lanes = terms.getResult(0);
        }
        blockBuilder.create<vector::StoreOp>(blockLoc, lanes, shape.output,
                                             ValueRange{shape.rowIndex, blockStart});
        blockBuilder.create<scf::YieldOp>(blockLoc);
      });

  // A fully covered ordered loop is erased rather than left dead: its body
  // would still record a spend the audit can never observe.
  if (batchedColumns == shape.columnCount) {
    loop.erase();
    return;
  }
  loop.getLowerBoundMutable().assign(batchedEnd);
}

class VectorizeOndspFpFilterOutputsPass final
    : public ondrix::impl::VectorizeOndspFpFilterOutputsBase<VectorizeOndspFpFilterOutputsPass> {
public:
  using ondrix::impl::VectorizeOndspFpFilterOutputsBase<
      VectorizeOndspFpFilterOutputsPass>::VectorizeOndspFpFilterOutputsBase;

  void runOnOperation() override {
    if (vectorWidth <= 1) {
      getOperation().emitError("vector-width must be greater than one");
      signalPassFailure();
      return;
    }
    if (vectorWidth > kMaxVectorWidth) {
      getOperation().emitError("vector-width must not exceed ") << kMaxVectorWidth;
      signalPassFailure();
      return;
    }

    // Collect first: the batched loop this pass creates must never be offered
    // to the matcher, and the ordered loop is mutated in place.
    SmallVector<scf::ForOp> candidates;
    getOperation().walk([&](scf::ForOp loop) { candidates.push_back(loop); });

    OpBuilder builder(&getContext());
    for (scf::ForOp loop : candidates) {
      if (FailureOr<FpFilterLoopShape> shape = matchFpFilterLoop(loop, vectorWidth);
          succeeded(shape)) {
        batchFpFilterOutputs(*shape, vectorWidth, supportsVectorFma, builder);
        continue;
      }
      if (FailureOr<FpWindowSumLoopShape> shape = matchFpWindowSumLoop(loop, vectorWidth);
          succeeded(shape)) {
        batchFpWindowSumOutputs(*shape, vectorWidth, builder);
        continue;
      }
      if (FailureOr<FpColumnTileLoopShape> shape = matchFpColumnTileLoop(loop, vectorWidth);
          succeeded(shape))
        batchFpColumnTiles(*shape, vectorWidth, supportsVectorFma, builder);
    }
    ondrix::ondsp::summarizeFastPermissions(getOperation());
  }
};

} // namespace

std::unique_ptr<Pass> ondrix::createVectorizeOndspFpFilterOutputsPass() {
  return std::make_unique<VectorizeOndspFpFilterOutputsPass>();
}

std::unique_ptr<Pass> ondrix::createVectorizeOndspFpFilterOutputsPass(
    const ondrix::VectorizeOndspFpFilterOutputsOptions &options) {
  return std::make_unique<VectorizeOndspFpFilterOutputsPass>(options);
}
