#include "ondrix/Conversion/OndspVectorization/OndspVectorization.h"
#include "ondrix/Conversion/Utils/FixedPointDomainUtils.h"
#include "ondrix/Conversion/Utils/MemRefLayoutUtils.h"

#include "ondrix/Dialect/ondsp/IR/OndspDialect.h"
#include "ondrix/Dialect/ondsp/IR/OndspOps.h"
#include "ondrix/Dialect/ondsp/IR/OndspSemantics.h"
#include "ondrix/Dialect/ondsp/IR/OndspTypes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/SmallVector.h"

#include <limits>
#include <optional>

namespace ondrix {
#define GEN_PASS_DEF_VECTORIZEONDSPFIXEDDECIMATEOUTPUTS
#include "ondrix/Conversion/Passes.h.inc"
} // namespace ondrix

using namespace mlir;

namespace {

/// Decimation factor this slice batches. A factor of two makes the contiguous
/// span covering W outputs exactly 2W elements, so one load plus one even-lane
/// shuffle produces the whole batch. Other factors need a different extraction
/// and stay on the ordered path.
constexpr int64_t kSupportedFactor = 2;

/// Tap count above which the batched body is not emitted. The taps are
/// unrolled so that the multi-lane accumulator stays in vector registers, and
/// an unbounded unroll would trade the ordered loop's compact code for an
/// arbitrarily large body. A longer filter keeps the ordered schedule.
constexpr int64_t kMaxUnrolledTaps = 256;

/// Largest accepted batch width. It bounds the lane count the accumulator type
/// receives and every span index derived from it, well below any width a target
/// register file makes sense for.
constexpr int64_t kMaxVectorWidth = 4096;

/// Everything the matcher recovered from one bufferized decimation loop. The
/// loop is only rewritten when every field is present and consistent, so the
/// matcher never leaves a partially understood loop behind.
struct DecimateLoopShape {
  scf::ForOp loop;
  /// Number of outputs the ordered loop computes.
  int64_t outputLength = 0;
  /// Static coefficient count, which is also the window length.
  int64_t coefficientLength = 0;
  /// Memref the window views, indexed directly by the batched body.
  Value input;
  /// Coefficient sequence exactly as the ordered reduction indexed it.
  Value coefficients;
  Value output;
  ondrix::ondsp::AccType accumulator;
  ondrix::ondsp::FixedAttr numeric;
  ondrix::ondsp::ProductAttr product;
  /// Destination policy of the ordered export, reused verbatim per lane.
  ondrix::ondsp::FixedAttr destination;
  ondrix::ondsp::RoundingMode rounding = ondrix::ondsp::RoundingMode::NearestEven;
  ondrix::ondsp::OverflowMode overflow = ondrix::ondsp::OverflowMode::Saturate;
};

/// Rank-1 memref whose single dimension is contiguous and whose memory space
/// the Vector to LLVM lowering accepts.
bool isBatchableRankOneMemRef(Value value) {
  auto type = dyn_cast<MemRefType>(value.getType());
  return type && type.getRank() == 1 && isLastMemrefDimUnitStride(type) &&
         ondrix::conversion::hasDefaultLLVMVectorMemorySpace(type);
}

/// Skips the layout-erasing casts bufferization inserts between a producer and
/// a dynamic-shaped consumer. The cast preserves the element sequence, so the
/// batched body may index the pre-cast value instead.
Value lookThroughMemRefCasts(Value value) {
  while (auto cast = value.getDefiningOp<memref::CastOp>())
    value = cast.getSource();
  return value;
}

/// Returns the multiplier when `value` is `iv * constant` in either operand
/// order, and nothing otherwise.
std::optional<int64_t> matchInductionVariableScale(Value value, Value inductionVariable) {
  auto multiply = value.getDefiningOp<arith::MulIOp>();
  if (!multiply)
    return std::nullopt;
  if (multiply.getLhs() == inductionVariable)
    return getConstantIntValue(multiply.getRhs());
  if (multiply.getRhs() == inductionVariable)
    return getConstantIntValue(multiply.getLhs());
  return std::nullopt;
}

/// Static length of a rank-1 memref sequence, resolved through the casts that
/// erase it. A dynamic length that no producer pins down is refused.
std::optional<int64_t> getStaticRankOneLength(Value value) {
  auto type = dyn_cast<MemRefType>(lookThroughMemRefCasts(value).getType());
  if (!type || type.getRank() != 1 || type.isDynamicDim(0))
    return std::nullopt;
  return type.getDimSize(0);
}

/// Matches the exact loop shape the decimation bufferization emits: one output
/// per iteration, a unit-stride window at offset `m * factor`, a zeroed
/// accumulator, one ordered memref reduction, one export, and one store. Any
/// other body, any dynamic extent the batched body would need, and any
/// accumulator profile the per-lane arithmetic does not implement all fail
/// closed, leaving the ordered loop untouched.
FailureOr<DecimateLoopShape> matchDecimateLoop(scf::ForOp loop, int64_t vectorWidth) {
  if (!loop.getInitArgs().empty())
    return failure();

  std::optional<int64_t> lowerBound = getConstantIntValue(loop.getLowerBound());
  std::optional<int64_t> upperBound = getConstantIntValue(loop.getUpperBound());
  std::optional<int64_t> step = getConstantIntValue(loop.getStep());
  if (!lowerBound || !upperBound || !step || *lowerBound != 0 || *step != 1 || *upperBound <= 0)
    return failure();

  Block &body = *loop.getBody();
  Value inductionVariable = loop.getInductionVar();

  // Walk the body in order and require exactly the expected operations. A
  // stricter match than necessary is deliberate: an unrecognized operation
  // could carry a side effect or a second use that batching would reorder.
  SmallVector<Operation *> operations;
  for (Operation &operation : body.without_terminator()) {
    // Casts erase the window layout on the way into a dynamic-shaped
    // reduction. They carry no arithmetic, so they are skipped rather than
    // counted, and the values they wrap are resolved back to their sources.
    if (isa<memref::CastOp>(operation))
      continue;
    operations.push_back(&operation);
  }
  if (operations.size() != 6)
    return failure();

  auto offset = dyn_cast<arith::MulIOp>(operations[0]);
  auto window = dyn_cast<memref::SubViewOp>(operations[1]);
  auto zero = dyn_cast<ondrix::ondsp::AccZeroOp>(operations[2]);
  auto reduce = dyn_cast<ondrix::ondsp::ReduceMacOp>(operations[3]);
  auto exportOp = dyn_cast<ondrix::ondsp::AccExportOp>(operations[4]);
  auto store = dyn_cast<memref::StoreOp>(operations[5]);
  if (!offset || !window || !zero || !reduce || !exportOp || !store)
    return failure();

  std::optional<int64_t> factor =
      matchInductionVariableScale(offset.getResult(), inductionVariable);
  if (!factor || *factor != kSupportedFactor)
    return failure();

  if (exportOp.getAcc() != reduce.getResult())
    return failure();
  if (store.getValueToStore() != exportOp.getResult() || store.getIndices().size() != 1 ||
      store.getIndices().front() != inductionVariable)
    return failure();

  // Single-use chains only. A second consumer of the accumulator, the reduced
  // value, or the exported sample would survive the rewrite unserved. The
  // window is allowed one extra user because a layout-erasing cast counts as
  // one; the cast's own result is then required to be single use.
  if (!zero.getAcc().hasOneUse() || !reduce.getResult().hasOneUse() ||
      !exportOp.getResult().hasOneUse())
    return failure();
  if (reduce.getInitial() != zero.getAcc())
    return failure();
  if (!window.getResult().hasOneUse() || !reduce.getLhs().hasOneUse())
    return failure();

  // The reduction must consume this iteration's window, possibly through the
  // layout-erasing cast bufferization inserts.
  if (lookThroughMemRefCasts(reduce.getLhs()) != window.getResult())
    return failure();

  // The window must be a plain unit-stride rank-1 slice at offset `m * factor`.
  if (window.getType().getRank() != 1 || window.getMixedOffsets().size() != 1 ||
      window.getMixedSizes().size() != 1 || window.getMixedStrides().size() != 1)
    return failure();
  auto windowOffset = window.getMixedOffsets().front().dyn_cast<Value>();
  if (!windowOffset || windowOffset != offset.getResult())
    return failure();
  std::optional<int64_t> windowStride = getConstantIntValue(window.getMixedStrides().front());
  std::optional<int64_t> windowLength = getConstantIntValue(window.getMixedSizes().front());
  if (!windowStride || *windowStride != 1 || !windowLength || *windowLength <= 0 ||
      *windowLength > kMaxUnrolledTaps)
    return failure();

  DecimateLoopShape shape;
  shape.loop = loop;
  shape.outputLength = *upperBound;
  shape.coefficientLength = *windowLength;
  shape.input = window.getSource();
  shape.coefficients = reduce.getRhs();
  shape.output = store.getMemRef();
  shape.destination = exportOp.getDst();
  shape.rounding = exportOp.getRounding();
  shape.overflow = exportOp.getOverflow();

  // The coefficient sequence is indexed by the tap index in both schedules, so
  // its length must be statically known and equal to the window length.
  std::optional<int64_t> coefficientLength = getStaticRankOneLength(shape.coefficients);
  if (!coefficientLength || *coefficientLength != shape.coefficientLength)
    return failure();
  shape.coefficients = lookThroughMemRefCasts(shape.coefficients);

  // Layouts the batched body can address contiguously.
  if (!isBatchableRankOneMemRef(shape.input) || !isBatchableRankOneMemRef(shape.coefficients) ||
      !isBatchableRankOneMemRef(shape.output))
    return failure();
  // The batched loop is placed immediately before the ordered one, so every
  // memref it addresses must be defined outside the ordered body.
  if (shape.coefficients.getParentBlock() == &body || shape.input.getParentBlock() == &body ||
      shape.output.getParentBlock() == &body)
    return failure();

  // The rewrite moves the W stores of a block past all K tap loads of that
  // block, so it is only sound when the three sequences are distinct storage;
  // the refusal set and the run-time precondition are in the pass description.
  if (ondrix::conversion::mayShareStorage(shape.input, shape.output) ||
      ondrix::conversion::mayShareStorage(shape.coefficients, shape.output) ||
      ondrix::conversion::mayShareStorage(shape.input, shape.coefficients))
    return failure();

  // The accumulator profile must be one the per-lane arithmetic implements, and
  // it must still be single-lane: widening an already batched accumulator would
  // change what a lane means.
  auto accumulator = dyn_cast<ondrix::ondsp::AccType>(reduce.getInitial().getType());
  auto numeric = dyn_cast<ondrix::ondsp::FixedAttr>(reduce.getNumeric());
  if (!accumulator || !numeric || !reduce.getProduct())
    return failure();
  if (!ondrix::ondsp::isSingleLaneAccumulator(accumulator))
    return failure();
  if (!ondrix::conversion::isSupportedFixedScalarMacDomain(accumulator, numeric,
                                                           *reduce.getProduct()))
    return failure();
  shape.accumulator = accumulator;
  shape.numeric = numeric;
  shape.product = *reduce.getProduct();

  auto storage = dyn_cast<IntegerType>(numeric.getStorage());
  auto destinationStorage = dyn_cast<IntegerType>(exportOp.getDst().getStorage());
  if (!storage || !destinationStorage)
    return failure();
  if (cast<MemRefType>(shape.input.getType()).getElementType() != storage ||
      cast<MemRefType>(shape.coefficients.getType()).getElementType() != storage ||
      cast<MemRefType>(shape.output.getType()).getElementType() != destinationStorage)
    return failure();

  // The batched span must stay inside the input. Every index below is bounded
  // by the static input length, so bound that first and the address arithmetic
  // cannot overflow.
  auto inputType = cast<MemRefType>(shape.input.getType());
  if (inputType.isDynamicDim(0))
    return failure();
  int64_t inputLength = inputType.getDimSize(0);
  if (inputLength <= 0 ||
      inputLength > std::numeric_limits<int64_t>::max() / (kSupportedFactor + 1))
    return failure();
  if (shape.outputLength > inputLength || shape.coefficientLength > inputLength)
    return failure();

  // Full blocks are restricted to those whose contiguous `factor * W` load
  // stays inside the input: the span covering outputs `m .. m + W - 1` at the
  // last tap ends one element past the last element the ordered schedule
  // reads, so the block containing the final output is always left to the
  // ordered loop. With `W` not dividing the output length this costs nothing,
  // and it is what keeps the batched load in bounds when it does.
  int64_t fullBlocks = (shape.outputLength - 1) / vectorWidth;
  if (fullBlocks < 1)
    return failure();

  // Independently of the argument above, pin the actual extent.
  int64_t lastBlockStart = (fullBlocks - 1) * vectorWidth;
  int64_t lastLoadBase = lastBlockStart * kSupportedFactor + shape.coefficientLength - 1;
  if (lastLoadBase + kSupportedFactor * vectorWidth > inputLength)
    return failure();

  return shape;
}

/// Replaces the leading full blocks of an ordered decimation loop with a
/// batched loop over `vectorWidth` outputs at a time and moves the ordered
/// loop's lower bound past them. The ordered body is not touched, so the
/// remaining outputs keep exactly the schedule they had.
void batchDecimateOutputs(const DecimateLoopShape &shape, int64_t vectorWidth, OpBuilder &builder) {
  scf::ForOp loop = shape.loop;
  Location loc = loop.getLoc();
  MLIRContext *context = builder.getContext();
  int64_t fullBlocks = (shape.outputLength - 1) / vectorWidth;
  int64_t batchedOutputs = fullBlocks * vectorWidth;

  auto laneAccumulator = ondrix::ondsp::AccType::get(
      context, shape.accumulator.getStorage(), shape.accumulator.getFrac(),
      shape.accumulator.getSignedness(), shape.accumulator.getUpdateOverflow(),
      static_cast<unsigned>(vectorWidth));
  auto storage = cast<IntegerType>(shape.numeric.getStorage());
  auto spanType = VectorType::get({kSupportedFactor * vectorWidth}, storage);
  auto sampleType =
      VectorType::get({vectorWidth}, cast<IntegerType>(shape.destination.getStorage()));

  // Phase-zero decimation keeps every `factor`-th element of the span.
  SmallVector<int64_t> evenLanes;
  for (int64_t lane = 0; lane < vectorWidth; ++lane)
    evenLanes.push_back(lane * kSupportedFactor);

  builder.setInsertionPoint(loop);
  Value zeroIndex = builder.create<arith::ConstantIndexOp>(loc, 0);
  Value factorIndex = builder.create<arith::ConstantIndexOp>(loc, kSupportedFactor);
  Value batchedEnd = builder.create<arith::ConstantIndexOp>(loc, batchedOutputs);
  Value batchStep = builder.create<arith::ConstantIndexOp>(loc, vectorWidth);
  SmallVector<Value> tapIndices;
  for (int64_t tap = 0; tap < shape.coefficientLength; ++tap)
    tapIndices.push_back(builder.create<arith::ConstantIndexOp>(loc, tap));

  builder.create<scf::ForOp>(
      loc, zeroIndex, batchedEnd, batchStep, ValueRange{},
      [&](OpBuilder &blockBuilder, Location blockLoc, Value blockStart, ValueRange) {
        Value accumulator =
            blockBuilder.create<ondrix::ondsp::AccZeroOp>(blockLoc, laneAccumulator);
        Value windowBase = blockBuilder.create<arith::MulIOp>(blockLoc, blockStart, factorIndex);
        // Taps are emitted in increasing order and each lane folds its own
        // products in that order, so this is the declared ordered update for
        // every one of the W outputs. The taps are unrolled rather than looped
        // because a loop-carried multi-lane accumulator forces the backend to
        // legalize its phi lane by lane, which spills exactly the independence
        // the batching exists to exploit.
        for (int64_t tap = 0; tap < shape.coefficientLength; ++tap) {
          Value base =
              tap == 0 ? windowBase
                       : blockBuilder.create<arith::AddIOp>(blockLoc, windowBase, tapIndices[tap]);
          Value span = blockBuilder.create<vector::LoadOp>(blockLoc, spanType, shape.input,
                                                           ValueRange{base});
          Value values = blockBuilder.create<vector::ShuffleOp>(blockLoc, span, span, evenLanes);
          Value coefficient = blockBuilder.create<memref::LoadOp>(blockLoc, shape.coefficients,
                                                                  ValueRange{tapIndices[tap]});
          accumulator = blockBuilder.create<ondrix::ondsp::MacOp>(blockLoc, laneAccumulator,
                                                                  accumulator, values, coefficient,
                                                                  shape.numeric, shape.product);
        }
        Value samples = blockBuilder.create<ondrix::ondsp::AccExportOp>(
            blockLoc, sampleType, accumulator, shape.destination, shape.rounding, shape.overflow);
        blockBuilder.create<vector::StoreOp>(blockLoc, samples, shape.output,
                                             ValueRange{blockStart});
        blockBuilder.create<scf::YieldOp>(blockLoc);
      });

  // The ordered loop keeps its body and now starts at the first output the
  // batched loop did not produce.
  loop.getLowerBoundMutable().assign(batchedEnd);
}

class VectorizeOndspFixedDecimateOutputsPass final
    : public ondrix::impl::VectorizeOndspFixedDecimateOutputsBase<
          VectorizeOndspFixedDecimateOutputsPass> {
public:
  using ondrix::impl::VectorizeOndspFixedDecimateOutputsBase<
      VectorizeOndspFixedDecimateOutputsPass>::VectorizeOndspFixedDecimateOutputsBase;

  void runOnOperation() override {
    if (vectorWidth <= 1) {
      getOperation().emitError("vector-width must be greater than one");
      signalPassFailure();
      return;
    }
    // The width becomes the accumulator's lane count, which is an `unsigned`.
    // Without an upper bound a width above the unsigned range would truncate on
    // the way into the type, and a multiple of 2^32 would truncate to zero
    // lanes — a value the type verifier rejects but the unchecked builder the
    // rewrite uses does not see.
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
      FailureOr<DecimateLoopShape> shape = matchDecimateLoop(loop, vectorWidth);
      if (failed(shape))
        continue;
      batchDecimateOutputs(*shape, vectorWidth, builder);
    }
  }
};

} // namespace

std::unique_ptr<Pass> ondrix::createVectorizeOndspFixedDecimateOutputsPass() {
  return std::make_unique<VectorizeOndspFixedDecimateOutputsPass>();
}

std::unique_ptr<Pass> ondrix::createVectorizeOndspFixedDecimateOutputsPass(
    const VectorizeOndspFixedDecimateOutputsOptions &options) {
  return std::make_unique<VectorizeOndspFixedDecimateOutputsPass>(options);
}
