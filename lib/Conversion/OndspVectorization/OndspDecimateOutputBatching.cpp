#include "ondrix/Analysis/FixedPointPrefixRangeAnalysis.h"
#include "ondrix/Conversion/OndspVectorization/OndspVectorization.h"
#include "ondrix/Conversion/Utils/FixedPointDomainUtils.h"
#include "ondrix/Conversion/Utils/MemRefLayoutUtils.h"
#include "ondrix/Conversion/Utils/ReductionUtils.h"

#include "ondrix/Dialect/ondsp/IR/OndspDialect.h"
#include "ondrix/Dialect/ondsp/IR/OndspOps.h"
#include "ondrix/Dialect/ondsp/IR/OndspSemantics.h"
#include "ondrix/Dialect/ondsp/IR/OndspTypes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/MathExtras.h"

#include <limits>
#include <optional>

namespace ondrix {
#define GEN_PASS_DEF_VECTORIZEONDSPFIXEDDECIMATEOUTPUTS
#include "ondrix/Conversion/Passes.h.inc"
} // namespace ondrix

using namespace mlir;

namespace {

/// Window strides this pass batches: one (a sliding window, the FIR and
/// convolution shape) loads the W samples of a tap directly; two (phase-zero
/// decimation) loads a 2W span and keeps its even lanes. Other strides need a
/// different extraction and stay on the ordered path.
constexpr int64_t kMaxSupportedFactor = 2;

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
  /// Window stride between consecutive outputs, one or two.
  int64_t factor = 1;
  /// Number of full blocks the batched loop computes.
  int64_t fullBlocks = 0;
  /// Number of outputs the ordered loop computes.
  int64_t outputLength = 0;
  /// Static coefficient count, which is also the window length.
  int64_t coefficientLength = 0;
  /// Memref the window views, indexed directly by the batched body.
  Value input;
  /// Coefficient sequence exactly as the ordered reduction indexed it.
  Value coefficients;
  /// The same sequence as compile-time constants when its storage is an
  /// immutable global, so each tap reads an immediate instead of a load.
  std::optional<SmallVector<llvm::APInt>> constantCoefficients;
  Value output;
  ondrix::ondsp::AccType accumulator;
  ondrix::ondsp::FixedAttr numeric;
  ondrix::ondsp::ProductAttr product;
  /// Destination policy of the ordered export, reused verbatim per lane.
  ondrix::ondsp::FixedAttr destination;
  ondrix::ondsp::RoundingMode rounding = ondrix::ondsp::RoundingMode::NearestEven;
  ondrix::ondsp::OverflowMode overflow = ondrix::ondsp::OverflowMode::Saturate;
  /// Set when a constant coefficient sequence certifies that no ordered prefix
  /// reaches the accumulator rail, so the lanes may update wrapping.
  bool certifiedWrap = false;
  /// On wrapping lanes, the number of consecutive taps whose products are
  /// certified to sum within i32, so they accumulate in an i32 lane group
  /// before joining the declared accumulator; zero keeps every tap direct.
  int64_t termGroup = 0;
};

/// The coefficient memref of one tap read: rank-1, default memory space, unit
/// or reversed stride. Only scalar loads address it, so the stride is free.
bool isTapReadableRankOneMemRef(Value value) {
  auto type = dyn_cast<MemRefType>(value.getType());
  if (!type || type.getRank() != 1 || type.isDynamicDim(0) ||
      !ondrix::conversion::hasDefaultLLVMVectorMemorySpace(type))
    return false;
  SmallVector<int64_t> strides;
  int64_t offset = 0;
  return succeeded(getStridesAndOffset(type, strides, offset)) && strides.size() == 1 &&
         (strides[0] == 1 || strides[0] == -1);
}

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
  // A stride-two window is offset by `iv * 2` computed in the body; a
  // stride-one window is offset by the induction variable itself.
  if (operations.size() != 5 && operations.size() != 6)
    return failure();
  int64_t factor = 1;
  Value windowOffsetValue = inductionVariable;
  if (operations.size() == 6) {
    auto offset = dyn_cast<arith::MulIOp>(operations[0]);
    if (!offset)
      return failure();
    std::optional<int64_t> scale =
        matchInductionVariableScale(offset.getResult(), inductionVariable);
    if (!scale || *scale != kMaxSupportedFactor)
      return failure();
    factor = *scale;
    windowOffsetValue = offset.getResult();
    operations.erase(operations.begin());
  }

  auto window = dyn_cast<memref::SubViewOp>(operations[0]);
  auto zero = dyn_cast<ondrix::ondsp::AccZeroOp>(operations[1]);
  auto reduce = dyn_cast<ondrix::ondsp::ReduceMacOp>(operations[2]);
  auto exportOp = dyn_cast<ondrix::ondsp::AccExportOp>(operations[3]);
  auto store = dyn_cast<memref::StoreOp>(operations[4]);
  if (!window || !zero || !reduce || !exportOp || !store)
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
  if (!windowOffset || windowOffset != windowOffsetValue)
    return failure();
  std::optional<int64_t> windowStride = getConstantIntValue(window.getMixedStrides().front());
  std::optional<int64_t> windowLength = getConstantIntValue(window.getMixedSizes().front());
  if (!windowStride || *windowStride != 1 || !windowLength || *windowLength <= 0 ||
      *windowLength > kMaxUnrolledTaps)
    return failure();

  DecimateLoopShape shape;
  shape.loop = loop;
  shape.factor = factor;
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

  // Layouts the batched body can address: contiguous samples and outputs, and
  // a coefficient sequence read one scalar at a time.
  if (!isBatchableRankOneMemRef(shape.input) || !isTapReadableRankOneMemRef(shape.coefficients) ||
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
      inputLength > std::numeric_limits<int64_t>::max() / (kMaxSupportedFactor + 1))
    return failure();
  if (shape.outputLength > inputLength || shape.coefficientLength > inputLength)
    return failure();

  // Full blocks are those whose contiguous `factor * W` load at the last tap
  // stays inside the input; a stride-two span ends one element past the last
  // element the ordered schedule reads, so its final block stays ordered.
  int64_t fullBlocks = shape.outputLength / vectorWidth;
  auto lastLoadEnd = [&](int64_t blocks) {
    return (blocks - 1) * vectorWidth * factor + shape.coefficientLength - 1 + factor * vectorWidth;
  };
  while (fullBlocks >= 1 && lastLoadEnd(fullBlocks) > inputLength)
    --fullBlocks;
  if (fullBlocks < 1)
    return failure();
  shape.fullBlocks = fullBlocks;

  // A saturating profile keeps its clamp per lane unless a constant coefficient
  // sequence certifies that no ordered prefix reaches the rail; a wrapping
  // profile needs no rail certificate, only the group one for narrow terms.
  shape.constantCoefficients = ondrix::conversion::getConstantCoefficientsInReadOrder(
      shape.coefficients, shape.coefficientLength, kMaxUnrolledTaps);
  const std::optional<SmallVector<llvm::APInt>> &constants = shape.constantCoefficients;
  if (!constants)
    return shape;
  if (accumulator.getUpdateOverflow() == ondrix::ondsp::OverflowMode::Saturate)
    shape.certifiedWrap = succeeded(
        ondrix::analysis::FixedPointPrefixRangePlanner::proveOrderedZeroSeededConstantReduction(
            reduce, *constants));
  if (shape.certifiedWrap || accumulator.getUpdateOverflow() == ondrix::ondsp::OverflowMode::Wrap)
    shape.termGroup = ondrix::analysis::FixedPointPrefixRangePlanner::largestCertifiedTermGroup(
        reduce, *constants, /*termWidth=*/32);

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
  int64_t batchedOutputs = shape.fullBlocks * vectorWidth;

  auto laneAccumulator =
      ondrix::ondsp::AccType::get(context, shape.accumulator.getStorage(),
                                  shape.accumulator.getFrac(), shape.accumulator.getSignedness(),
                                  shape.certifiedWrap ? ondrix::ondsp::OverflowMode::Wrap
                                                      : shape.accumulator.getUpdateOverflow(),
                                  static_cast<unsigned>(vectorWidth));
  auto storage = cast<IntegerType>(shape.numeric.getStorage());
  auto spanType = VectorType::get({shape.factor * vectorWidth}, storage);
  // A certified tap group accumulates in i32 lanes and joins the declared
  // accumulator as one term; both are wrapping, so the composition is exact.
  int64_t groupedTaps =
      shape.termGroup >= 2 ? (shape.coefficientLength / shape.termGroup) * shape.termGroup : 0;
  ondrix::ondsp::AccType groupAccumulator;
  ondrix::ondsp::FixedAttr groupTermNumeric;
  VectorType groupTermType;
  if (groupedTaps > 0) {
    groupAccumulator = ondrix::ondsp::AccType::get(
        context, builder.getI32Type(), shape.accumulator.getFrac(),
        shape.accumulator.getSignedness(), ondrix::ondsp::OverflowMode::Wrap,
        static_cast<unsigned>(vectorWidth));
    groupTermNumeric =
        ondrix::ondsp::FixedAttr::get(context, shape.accumulator.getSignedness(),
                                      builder.getI32Type(), shape.accumulator.getFrac());
    groupTermType = VectorType::get({vectorWidth}, builder.getI32Type());
  }
  auto sampleType =
      VectorType::get({vectorWidth}, cast<IntegerType>(shape.destination.getStorage()));

  // Phase-zero decimation keeps every `factor`-th element of the span.
  SmallVector<int64_t> evenLanes;
  for (int64_t lane = 0; lane < vectorWidth; ++lane)
    evenLanes.push_back(lane * shape.factor);

  builder.setInsertionPoint(loop);
  Value zeroIndex = builder.create<arith::ConstantIndexOp>(loc, 0);
  Value factorIndex = builder.create<arith::ConstantIndexOp>(loc, shape.factor);
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
        Value windowBase =
            shape.factor == 1
                ? blockStart
                : blockBuilder.create<arith::MulIOp>(blockLoc, blockStart, factorIndex).getResult();
        // Taps are emitted in increasing order and each lane folds its own
        // products in that order, so this is the declared ordered update for
        // every one of the W outputs. The taps are unrolled rather than looped
        // because a loop-carried multi-lane accumulator forces the backend to
        // legalize its phi lane by lane, which spills exactly the independence
        // the batching exists to exploit.
        Value group;
        for (int64_t tap = 0; tap < shape.coefficientLength; ++tap) {
          bool grouped = tap < groupedTaps;
          if (grouped && tap % shape.termGroup == 0)
            group = blockBuilder.create<ondrix::ondsp::AccZeroOp>(blockLoc, groupAccumulator);
          Value base =
              tap == 0 ? windowBase
                       : blockBuilder.create<arith::AddIOp>(blockLoc, windowBase, tapIndices[tap]);
          Value span = blockBuilder.create<vector::LoadOp>(blockLoc, spanType, shape.input,
                                                           ValueRange{base});
          Value values = span;
          if (shape.factor != 1)
            values = blockBuilder.create<vector::ShuffleOp>(blockLoc, span, span, evenLanes);
          Value coefficient =
              shape.constantCoefficients
                  ? blockBuilder
                        .create<arith::ConstantOp>(
                            blockLoc, IntegerAttr::get(shape.numeric.getStorage(),
                                                       (*shape.constantCoefficients)[tap]))
                        .getResult()
                  : blockBuilder
                        .create<memref::LoadOp>(blockLoc, shape.coefficients,
                                                ValueRange{tapIndices[tap]})
                        .getResult();
          if (!grouped) {
            accumulator = blockBuilder.create<ondrix::ondsp::MacOp>(
                blockLoc, laneAccumulator, accumulator, values, coefficient, shape.numeric,
                shape.product);
            continue;
          }
          group = blockBuilder.create<ondrix::ondsp::MacOp>(
              blockLoc, groupAccumulator, group, values, coefficient, shape.numeric, shape.product);
          if (tap % shape.termGroup == shape.termGroup - 1) {
            Value term = blockBuilder.create<ondrix::ondsp::AccExportOp>(
                blockLoc, groupTermType, group, groupTermNumeric,
                ondrix::ondsp::RoundingMode::TowardNegative, ondrix::ondsp::OverflowMode::Wrap);
            accumulator = blockBuilder.create<ondrix::ondsp::AccAddTermOp>(
                blockLoc, laneAccumulator, accumulator, term, groupTermNumeric);
          }
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

//===----------------------------------------------------------------------===//
// Matrix outputs: one lane per output column
//===----------------------------------------------------------------------===//

/// A loop over `[0, extent)` with unit step and no iteration arguments.
bool isUnitStepLoop(scf::ForOp loop, int64_t extent) {
  std::optional<int64_t> lower = getConstantIntValue(loop.getLowerBound());
  std::optional<int64_t> upper = getConstantIntValue(loop.getUpperBound());
  std::optional<int64_t> step = getConstantIntValue(loop.getStep());
  return loop.getInitArgs().empty() && lower && upper && step && *lower == 0 && *step == 1 &&
         *upper == extent;
}

/// Operations of a block that carry its shape, with constants and the
/// layout-erasing casts skipped.
SmallVector<Operation *> getShapeOperations(Block &body) {
  SmallVector<Operation *> operations;
  for (Operation &op : body.without_terminator()) {
    if (matchPattern(&op, m_Constant()) || isa<memref::CastOp>(op))
      continue;
    operations.push_back(&op);
  }
  return operations;
}

/// Rank-2 memref with a static shape whose rows are contiguous, so a row
/// segment is one vector load.
bool isRowContiguousMatrix(Value value) {
  auto type = dyn_cast<MemRefType>(value.getType());
  if (!type || type.getRank() != 2 || !type.hasStaticShape() ||
      !ondrix::conversion::hasDefaultLLVMVectorMemorySpace(type))
    return false;
  SmallVector<int64_t> strides;
  int64_t offset = 0;
  return succeeded(getStridesAndOffset(type, strides, offset)) && strides[1] == 1;
}

/// Matches the two perfectly nested loops that transpose `source[k][c]` into
/// `packed[c][k]`, and returns the source matrix.
Value matchPackNest(scf::ForOp outer, Value packed) {
  auto packedType = cast<MemRefType>(packed.getType());
  if (!isUnitStepLoop(outer, packedType.getDimSize(0)))
    return nullptr;
  SmallVector<Operation *> outerBody = getShapeOperations(*outer.getBody());
  if (outerBody.size() != 1)
    return nullptr;
  auto inner = dyn_cast<scf::ForOp>(outerBody.front());
  if (!inner || !isUnitStepLoop(inner, packedType.getDimSize(1)))
    return nullptr;
  SmallVector<Operation *> innerBody = getShapeOperations(*inner.getBody());
  if (innerBody.size() != 2)
    return nullptr;
  auto load = dyn_cast<memref::LoadOp>(innerBody[0]);
  auto store = dyn_cast<memref::StoreOp>(innerBody[1]);
  if (!load || !store || store.getValueToStore() != load.getResult())
    return nullptr;
  Value column = outer.getInductionVar();
  Value row = inner.getInductionVar();
  if (!llvm::equal(load.getIndices(), ValueRange{row, column}) || store.getMemRef() != packed ||
      !llvm::equal(store.getIndices(), ValueRange{column, row}))
    return nullptr;
  Value source = load.getMemRef();
  auto sourceType = dyn_cast<MemRefType>(source.getType());
  SmallVector<int64_t> strides;
  int64_t offset = 0;
  if (!isRowContiguousMatrix(source) || failed(getStridesAndOffset(sourceType, strides, offset)) ||
      strides[0] != sourceType.getDimSize(1) ||
      sourceType.getDimSize(0) != packedType.getDimSize(1) ||
      sourceType.getDimSize(1) != packedType.getDimSize(0) ||
      sourceType.getElementType() != packedType.getElementType())
    return nullptr;
  return source;
}

/// One bufferized matrix-product column loop: every output column of one row
/// reduces the same row vector against a packed (transposed) column of the
/// right operand.
struct ColumnLoopShape {
  scf::ForOp loop;
  /// The transposing pack nest and the buffer it fills.
  scf::ForOp packNest;
  Value packed;
  /// The right operand as the caller passed it, `[k][column]`.
  Value matrix;
  /// The row of the left operand, contiguous and indexed by `k`.
  Value row;
  Value output;
  /// Leading store indices, defined outside the loop; the column is last.
  SmallVector<Value> outputLeadingIndices;
  int64_t columnCount = 0;
  int64_t innerCount = 0;
  ondrix::ondsp::AccType accumulator;
  ondrix::ondsp::FixedAttr numeric;
  ondrix::ondsp::ProductAttr product;
  ondrix::ondsp::FixedAttr destination;
  ondrix::ondsp::RoundingMode rounding;
  ondrix::ondsp::OverflowMode overflow;
};

/// The pack nest that fills `packed`, found among the buffer's users.
scf::ForOp findPackNest(Value packed, Value &matrix) {
  for (Operation *user : packed.getUsers()) {
    auto store = dyn_cast<memref::StoreOp>(user);
    if (!store)
      continue;
    auto inner = dyn_cast<scf::ForOp>(store->getParentOp());
    if (!inner)
      continue;
    auto outer = dyn_cast<scf::ForOp>(inner->getParentOp());
    if (!outer)
      continue;
    if (Value source = matchPackNest(outer, packed)) {
      matrix = source;
      return outer;
    }
  }
  return nullptr;
}

/// Whether `operation` and everything nested in it can only read memory.
bool onlyReadsMemory(Operation *operation) {
  if (isMemoryEffectFree(operation))
    return true;
  if (auto effects = dyn_cast<MemoryEffectOpInterface>(operation)) {
    SmallVector<MemoryEffects::EffectInstance> instances;
    effects.getEffects(instances);
    return llvm::all_of(instances, [](const MemoryEffects::EffectInstance &instance) {
      return isa<MemoryEffects::Read>(instance.getEffect());
    });
  }
  if (!operation->hasTrait<OpTrait::HasRecursiveMemoryEffects>())
    return false;
  for (Region &region : operation->getRegions())
    for (Block &block : region)
      for (Operation &nested : block)
        if (!onlyReadsMemory(&nested))
          return false;
  return true;
}

/// The batched loop reads the source matrix where the ordered loop read its
/// transposed copy, which is the same value only if nothing can write memory
/// between the pack nest and the column loop: the pack nest must come first in
/// a common block, and every operation from there to the loop, including the
/// loop's enclosing operations but not the loop itself, may only read.
bool isPackSnapshotIntact(scf::ForOp packNest, scf::ForOp loop) {
  Block *block = packNest->getBlock();
  Operation *ancestor = block->findAncestorOpInBlock(*loop);
  if (!ancestor || !packNest->isBeforeInBlock(ancestor))
    return false;
  for (Operation *between = packNest->getNextNode(); between != ancestor;
       between = between->getNextNode())
    if (!onlyReadsMemory(between))
      return false;
  if (ancestor == loop)
    return true;
  WalkResult result = ancestor->walk<WalkOrder::PreOrder>([&](Operation *operation) {
    if (operation == loop)
      return WalkResult::skip();
    if (operation->getNumRegions() > 0 && operation->hasTrait<OpTrait::HasRecursiveMemoryEffects>())
      return WalkResult::advance();
    return onlyReadsMemory(operation) ? WalkResult::advance() : WalkResult::interrupt();
  });
  return !result.wasInterrupted();
}

/// Matches the column loop the matrix-product bufferization emits: a row view
/// of the packed buffer selected by the induction variable, a zeroed
/// accumulator, one ordered reduction against a loop-invariant row, one export,
/// and one store whose last index is the induction variable.
FailureOr<ColumnLoopShape> matchColumnLoop(scf::ForOp loop) {
  std::optional<int64_t> upperBound = getConstantIntValue(loop.getUpperBound());
  if (!upperBound || *upperBound < 2 || !isUnitStepLoop(loop, *upperBound))
    return failure();
  Block &body = *loop.getBody();
  Value column = loop.getInductionVar();

  SmallVector<Operation *> operations = getShapeOperations(body);
  if (operations.size() != 5)
    return failure();
  auto view = dyn_cast<memref::SubViewOp>(operations[0]);
  auto zero = dyn_cast<ondrix::ondsp::AccZeroOp>(operations[1]);
  auto reduce = dyn_cast<ondrix::ondsp::ReduceMacOp>(operations[2]);
  auto exportOp = dyn_cast<ondrix::ondsp::AccExportOp>(operations[3]);
  auto store = dyn_cast<memref::StoreOp>(operations[4]);
  if (!view || !zero || !reduce || !exportOp || !store)
    return failure();
  if (reduce.getInitial() != zero.getAcc() || exportOp.getAcc() != reduce.getResult() ||
      store.getValueToStore() != exportOp.getResult())
    return failure();
  if (!zero.getAcc().hasOneUse() || !reduce.getResult().hasOneUse() ||
      !exportOp.getResult().hasOneUse() || !view.getResult().hasOneUse())
    return failure();
  if (lookThroughMemRefCasts(reduce.getRhs()) != view.getResult())
    return failure();

  // The view is row `column` of the packed buffer, all of its K elements.
  Value packed = view.getSource();
  auto packedType = dyn_cast<MemRefType>(packed.getType());
  if (!packedType || packedType.getRank() != 2 || !packedType.hasStaticShape() ||
      view.getType().getRank() != 1 || view.getMixedOffsets().size() != 2)
    return failure();
  auto viewColumn = view.getMixedOffsets()[0].dyn_cast<Value>();
  if (!viewColumn || viewColumn != column || !isConstantIntValue(view.getMixedOffsets()[1], 0) ||
      !isConstantIntValue(view.getMixedSizes()[0], 1) ||
      !isConstantIntValue(view.getMixedSizes()[1], packedType.getDimSize(1)) ||
      !isConstantIntValue(view.getMixedStrides()[0], 1) ||
      !isConstantIntValue(view.getMixedStrides()[1], 1))
    return failure();
  if (packedType.getDimSize(0) != *upperBound)
    return failure();

  ColumnLoopShape shape;
  shape.loop = loop;
  shape.packed = packed;
  shape.packNest = findPackNest(packed, shape.matrix);
  if (!shape.packNest || !isPackSnapshotIntact(shape.packNest, loop))
    return failure();
  shape.columnCount = *upperBound;
  shape.innerCount = packedType.getDimSize(1);
  if (shape.innerCount <= 0 || shape.innerCount > kMaxUnrolledTaps)
    return failure();

  // The reduced row is loop invariant, contiguous, and exactly K long.
  shape.row = lookThroughMemRefCasts(reduce.getLhs());
  std::optional<int64_t> rowLength = getStaticRankOneLength(shape.row);
  if (!rowLength || *rowLength != shape.innerCount || !isBatchableRankOneMemRef(shape.row) ||
      shape.row.getParentBlock() == &body)
    return failure();

  // The store addresses `output[..., column]` with a contiguous last dimension.
  shape.output = store.getMemRef();
  if (store.getIndices().empty() || store.getIndices().back() != column ||
      shape.output.getParentBlock() == &body)
    return failure();
  auto outputType = dyn_cast<MemRefType>(shape.output.getType());
  if (!outputType || !outputType.hasStaticShape() || !isLastMemrefDimUnitStride(outputType) ||
      !ondrix::conversion::hasDefaultLLVMVectorMemorySpace(outputType) ||
      outputType.getDimSize(outputType.getRank() - 1) != shape.columnCount)
    return failure();
  for (Value index : store.getIndices().drop_back()) {
    if (index.getParentBlock() == &body)
      return failure();
    shape.outputLeadingIndices.push_back(index);
  }

  // The rewrite reads the right operand directly and moves the stores of a
  // block past the loads of that block, so the three must be distinct storage.
  if (ondrix::conversion::mayShareStorage(shape.output, shape.matrix) ||
      ondrix::conversion::mayShareStorage(shape.output, shape.row) ||
      ondrix::conversion::mayShareStorage(shape.matrix, shape.row))
    return failure();

  auto accumulator = dyn_cast<ondrix::ondsp::AccType>(reduce.getInitial().getType());
  auto numeric = dyn_cast<ondrix::ondsp::FixedAttr>(reduce.getNumeric());
  if (!accumulator || !numeric || !reduce.getProduct() ||
      !ondrix::ondsp::isSingleLaneAccumulator(accumulator) ||
      !ondrix::conversion::isSupportedFixedScalarMacDomain(accumulator, numeric,
                                                           *reduce.getProduct()))
    return failure();
  auto storage = dyn_cast<IntegerType>(numeric.getStorage());
  auto destinationStorage = dyn_cast<IntegerType>(exportOp.getDst().getStorage());
  if (!storage || !destinationStorage ||
      cast<MemRefType>(shape.matrix.getType()).getElementType() != storage ||
      cast<MemRefType>(shape.row.getType()).getElementType() != storage ||
      outputType.getElementType() != destinationStorage)
    return failure();
  shape.accumulator = accumulator;
  shape.numeric = numeric;
  shape.product = *reduce.getProduct();
  shape.destination = exportOp.getDst();
  shape.rounding = exportOp.getRounding();
  shape.overflow = exportOp.getOverflow();
  return shape;
}

/// Rewrites the leading full blocks of a column loop to compute `lanes`
/// columns at a time: each term loads a contiguous row segment of the right
/// operand and broadcasts the row element, so no packed copy is read. A column
/// count below the vector width takes every column in one block of the next
/// power-of-two width; the surplus lanes read the following row while that
/// stays inside the matrix and zeros afterwards, and are never stored.
void batchColumnOutputs(const ColumnLoopShape &shape, int64_t vectorWidth, OpBuilder &builder) {
  scf::ForOp loop = shape.loop;
  Location loc = loop.getLoc();
  MLIRContext *context = builder.getContext();
  int64_t lanes = std::min(vectorWidth, shape.columnCount);
  int64_t storedLanes = lanes;
  if (shape.columnCount < vectorWidth)
    lanes = std::min(vectorWidth, static_cast<int64_t>(llvm::PowerOf2Ceil(shape.columnCount)));
  int64_t batchedColumns = (shape.columnCount / storedLanes) * storedLanes;
  // Elements the matrix holds; a wide load of the last rows would run past it.
  int64_t matrixElements = shape.innerCount * shape.columnCount;

  auto laneAccumulator = ondrix::ondsp::AccType::get(
      context, shape.accumulator.getStorage(), shape.accumulator.getFrac(),
      shape.accumulator.getSignedness(), shape.accumulator.getUpdateOverflow(),
      static_cast<unsigned>(lanes));
  auto storage = cast<IntegerType>(shape.numeric.getStorage());
  auto segmentType = VectorType::get({lanes}, storage);
  auto narrowSegmentType = VectorType::get({storedLanes}, storage);
  auto sampleType = VectorType::get({lanes}, cast<IntegerType>(shape.destination.getStorage()));
  SmallVector<int64_t> padMask;
  for (int64_t lane = 0; lane < lanes; ++lane)
    padMask.push_back(lane < storedLanes ? lane : storedLanes);

  builder.setInsertionPoint(loop);
  Value zeroIndex = builder.create<arith::ConstantIndexOp>(loc, 0);
  Value batchedEnd = builder.create<arith::ConstantIndexOp>(loc, batchedColumns);
  Value batchStep = builder.create<arith::ConstantIndexOp>(loc, storedLanes);
  SmallVector<Value> termIndices;
  for (int64_t term = 0; term < shape.innerCount; ++term)
    termIndices.push_back(builder.create<arith::ConstantIndexOp>(loc, term));

  builder.create<scf::ForOp>(
      loc, zeroIndex, batchedEnd, batchStep, ValueRange{},
      [&](OpBuilder &blockBuilder, Location blockLoc, Value blockStart, ValueRange) {
        Value accumulator =
            blockBuilder.create<ondrix::ondsp::AccZeroOp>(blockLoc, laneAccumulator);
        // Ascending k for every lane, which is the declared ordered update of
        // each of the `lanes` column reductions.
        Value zeroSegment;
        for (int64_t term = 0; term < shape.innerCount; ++term) {
          Value segment;
          if (term * shape.columnCount + lanes <= matrixElements) {
            segment = blockBuilder.create<vector::LoadOp>(
                blockLoc, segmentType, shape.matrix, ValueRange{termIndices[term], blockStart});
          } else {
            if (!zeroSegment)
              zeroSegment = blockBuilder.create<arith::ConstantOp>(
                  blockLoc,
                  DenseElementsAttr::get(narrowSegmentType, builder.getZeroAttr(storage)));
            Value narrow =
                blockBuilder.create<vector::LoadOp>(blockLoc, narrowSegmentType, shape.matrix,
                                                    ValueRange{termIndices[term], blockStart});
            segment =
                blockBuilder.create<vector::ShuffleOp>(blockLoc, narrow, zeroSegment, padMask);
          }
          Value element = blockBuilder.create<memref::LoadOp>(blockLoc, shape.row,
                                                              ValueRange{termIndices[term]});
          accumulator = blockBuilder.create<ondrix::ondsp::MacOp>(blockLoc, laneAccumulator,
                                                                  accumulator, segment, element,
                                                                  shape.numeric, shape.product);
        }
        Value samples = blockBuilder.create<ondrix::ondsp::AccExportOp>(
            blockLoc, sampleType, accumulator, shape.destination, shape.rounding, shape.overflow);
        if (storedLanes < lanes)
          samples = blockBuilder.create<vector::ExtractStridedSliceOp>(
              blockLoc, samples, ArrayRef<int64_t>{0}, ArrayRef<int64_t>{storedLanes},
              ArrayRef<int64_t>{1});
        SmallVector<Value> indices(shape.outputLeadingIndices);
        indices.push_back(blockStart);
        blockBuilder.create<vector::StoreOp>(blockLoc, samples, shape.output, indices);
        blockBuilder.create<scf::YieldOp>(blockLoc);
      });

  if (batchedColumns < shape.columnCount) {
    loop.getLowerBoundMutable().assign(batchedEnd);
    return;
  }
  // Every column is batched: the ordered loop is dead, and with it the packed
  // copy when nothing else reads it.
  loop.erase();
  Value packed = shape.packed;
  SmallVector<Operation *> otherUsers;
  for (Operation *user : packed.getUsers()) {
    if (isa<memref::DeallocOp>(user) || shape.packNest->isAncestor(user))
      continue;
    otherUsers.push_back(user);
  }
  if (!otherUsers.empty())
    return;
  scf::ForOp packNest = shape.packNest;
  packNest.erase();
  for (Operation *user : llvm::make_early_inc_range(packed.getUsers()))
    user->erase();
  if (Operation *alloc = packed.getDefiningOp(); alloc && isa<memref::AllocOp>(alloc))
    alloc->erase();
}

//===----------------------------------------------------------------------===//
// Constant-table outputs: one lane per output row
//===----------------------------------------------------------------------===//

/// One unrolled output of a constant-table product: an ordered reduction of a
/// shared input against an immutable constant row, exported, optionally
/// requantized, and stored at a static position.
struct ConstantRowOutput {
  ondrix::ondsp::AccZeroOp zero;
  ondrix::ondsp::ReduceMacOp reduce;
  ondrix::ondsp::AccExportOp exportOp;
  ondrix::ondsp::RoundShiftOp roundShift;
  memref::StoreOp store;
  int64_t position = 0;
  SmallVector<llvm::APInt> coefficients;
};

/// Attributes two outputs must share to be lanes of one block.
bool haveSameLaneContract(ConstantRowOutput &lhs, ConstantRowOutput &rhs) {
  if (lhs.reduce.getLhs() != rhs.reduce.getLhs() ||
      lhs.reduce.getInitial().getType() != rhs.reduce.getInitial().getType() ||
      lhs.reduce.getNumeric() != rhs.reduce.getNumeric() ||
      lhs.reduce.getProductAttr() != rhs.reduce.getProductAttr() ||
      lhs.exportOp.getDst() != rhs.exportOp.getDst() ||
      lhs.exportOp.getRounding() != rhs.exportOp.getRounding() ||
      lhs.exportOp.getOverflow() != rhs.exportOp.getOverflow() ||
      lhs.store.getMemRef() != rhs.store.getMemRef() ||
      static_cast<bool>(lhs.roundShift) != static_cast<bool>(rhs.roundShift))
    return false;
  return !lhs.roundShift || lhs.roundShift.getScale() == rhs.roundShift.getScale();
}

/// Matches the chain around `reduce`, or nothing. The zero seed may be shared
/// by several chains after common-subexpression elimination.
std::optional<ConstantRowOutput> matchConstantRowOutput(ondrix::ondsp::ReduceMacOp reduce) {
  ConstantRowOutput output;
  output.reduce = reduce;
  output.zero = reduce.getInitial().getDefiningOp<ondrix::ondsp::AccZeroOp>();
  if (!output.zero || !reduce.getResult().hasOneUse() || !reduce.getProduct())
    return std::nullopt;
  output.exportOp =
      dyn_cast<ondrix::ondsp::AccExportOp>(*output.reduce.getResult().getUsers().begin());
  if (!output.exportOp || !output.exportOp.getResult().hasOneUse())
    return std::nullopt;
  Value stored = output.exportOp.getResult();
  if (auto roundShift = dyn_cast<ondrix::ondsp::RoundShiftOp>(*stored.getUsers().begin())) {
    if (!roundShift.getResult().hasOneUse())
      return std::nullopt;
    output.roundShift = roundShift;
    stored = roundShift.getResult();
  }
  output.store = dyn_cast<memref::StoreOp>(*stored.getUsers().begin());
  if (!output.store || output.store.getIndices().size() != 1)
    return std::nullopt;
  std::optional<int64_t> position = getConstantIntValue(output.store.getIndices().front());
  if (!position)
    return std::nullopt;
  output.position = *position;

  // Every operation of the chain sits in the block of the reduction.
  Block *block = reduce->getBlock();
  for (Operation *op : {output.zero.getOperation(), output.exportOp.getOperation(),
                        output.roundShift ? output.roundShift.getOperation() : nullptr,
                        output.store.getOperation()})
    if (op && op->getBlock() != block)
      return std::nullopt;

  auto accumulator = dyn_cast<ondrix::ondsp::AccType>(reduce.getInitial().getType());
  auto numeric = dyn_cast<ondrix::ondsp::FixedAttr>(output.reduce.getNumeric());
  if (!accumulator || !numeric || !ondrix::ondsp::isSingleLaneAccumulator(accumulator) ||
      !ondrix::conversion::isSupportedFixedScalarMacDomain(accumulator, numeric,
                                                           *output.reduce.getProduct()))
    return std::nullopt;

  Value input = lookThroughMemRefCasts(output.reduce.getLhs());
  Value row = lookThroughMemRefCasts(output.reduce.getRhs());
  std::optional<int64_t> length = getStaticRankOneLength(input);
  if (!length || *length > kMaxUnrolledTaps || getStaticRankOneLength(row) != length ||
      !isBatchableRankOneMemRef(input) || !isTapReadableRankOneMemRef(row) ||
      !isBatchableRankOneMemRef(output.store.getMemRef()))
    return std::nullopt;
  if (ondrix::conversion::mayShareStorage(input, output.store.getMemRef()))
    return std::nullopt;
  auto storage = dyn_cast<IntegerType>(numeric.getStorage());
  if (!storage || cast<MemRefType>(input.getType()).getElementType() != storage ||
      cast<MemRefType>(row.getType()).getElementType() != storage)
    return std::nullopt;
  std::optional<SmallVector<llvm::APInt>> coefficients =
      ondrix::conversion::getConstantCoefficientsInReadOrder(row, *length, kMaxUnrolledTaps);
  if (!coefficients)
    return std::nullopt;
  output.coefficients = std::move(*coefficients);
  return output;
}

/// The store of the run that comes last in program order, when everything
/// between the first reduction and it is either a member of the run or free of
/// memory effects, so the run may be folded into one block placed there.
memref::StoreOp findRunEnd(MutableArrayRef<ConstantRowOutput> outputs) {
  DenseSet<Operation *> members;
  Operation *begin = nullptr;
  memref::StoreOp end;
  for (ConstantRowOutput &output : outputs) {
    for (Operation *op :
         {output.zero.getOperation(), output.reduce.getOperation(), output.exportOp.getOperation(),
          output.roundShift ? output.roundShift.getOperation() : nullptr,
          output.store.getOperation()})
      if (op)
        members.insert(op);
    if (!begin || output.reduce->isBeforeInBlock(begin))
      begin = output.reduce;
    if (!end || end->isBeforeInBlock(output.store))
      end = output.store;
  }
  for (Operation *op = begin; op != end.getOperation(); op = op->getNextNode())
    if (!members.contains(op) && !isMemoryEffectFree(op))
      return nullptr;
  return end;
}

/// The block's coefficients as one immutable column-major table: term `t`'s
/// column occupies `lanes` consecutive elements at `t * lanes`, so the loop
/// form reads one aligned machine vector per term.
Value createColumnTable(OpBuilder &builder, Location loc, ArrayRef<ConstantRowOutput> outputs,
                        int64_t length, IntegerType storage) {
  auto lanes = static_cast<int64_t>(outputs.size());
  SmallVector<llvm::APInt> values;
  values.reserve(length * lanes);
  for (int64_t term = 0; term < length; ++term)
    for (const ConstantRowOutput &output : outputs)
      values.push_back(output.coefficients[term]);
  auto tableType = MemRefType::get({length * lanes}, storage);
  auto initializer =
      DenseIntElementsAttr::get(RankedTensorType::get({length * lanes}, storage), values);

  auto module = builder.getInsertionBlock()->getParentOp()->getParentOfType<ModuleOp>();
  std::string symbol;
  for (int64_t ordinal = 0;; ++ordinal) {
    symbol = ("__ondrix_row_table_" + Twine(ordinal)).str();
    if (!SymbolTable::lookupSymbolIn(module, symbol))
      break;
  }
  {
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(module.getBody());
    builder.create<memref::GlobalOp>(loc, symbol, builder.getStringAttr("private"), tableType,
                                     initializer, /*constant=*/true,
                                     builder.getI64IntegerAttr(lanes * storage.getWidth() / 8));
  }
  return builder.create<memref::GetGlobalOp>(loc, tableType, symbol);
}

/// The block's one export, its optional requantization, and the one store
/// that replaces the run.
void emitBlockTail(OpBuilder &builder, Location loc, Value acc, ConstantRowOutput &first,
                   VectorType sampleType, int64_t lanes) {
  Value samples = builder.create<ondrix::ondsp::AccExportOp>(
      loc, sampleType, acc, first.exportOp.getDst(), first.exportOp.getRounding(),
      first.exportOp.getOverflow());
  if (first.roundShift) {
    auto storedType = VectorType::get({lanes}, first.roundShift.getResult().getType());
    samples = builder.create<ondrix::ondsp::RoundShiftOp>(loc, storedType, samples,
                                                          first.roundShift.getScale());
  }
  Value position = builder.create<arith::ConstantIndexOp>(loc, first.position);
  builder.create<vector::StoreOp>(loc, samples, first.store.getMemRef(), ValueRange{position});
}

void eraseBatchedOutputs(MutableArrayRef<ConstantRowOutput> outputs) {
  for (ConstantRowOutput &output : outputs) {
    output.store.erase();
    if (output.roundShift)
      output.roundShift.erase();
    output.exportOp.erase();
    output.reduce.erase();
    if (output.zero.getAcc().use_empty())
      output.zero.erase();
  }
}

/// Rewrites `outputs`, whose positions are consecutive, into one lane block.
void batchConstantRowOutputs(MutableArrayRef<ConstantRowOutput> outputs, OpBuilder &builder,
                             int64_t maxStraightLineCoefficients) {
  memref::StoreOp runEnd = findRunEnd(outputs);
  if (!runEnd)
    return;
  ConstantRowOutput &first = outputs.front();
  int64_t lanes = outputs.size();
  int64_t length = first.coefficients.size();
  Location loc = first.reduce.getLoc();
  MLIRContext *context = builder.getContext();
  auto accumulator = cast<ondrix::ondsp::AccType>(first.zero.getAcc().getType());
  auto numeric = cast<ondrix::ondsp::FixedAttr>(first.reduce.getNumeric());
  auto storage = cast<IntegerType>(numeric.getStorage());
  Value input = lookThroughMemRefCasts(first.reduce.getLhs());

  // A saturating profile drops its clamp only when every lane's row certifies
  // that no ordered prefix reaches the rail; the group width is the smallest
  // certified across the lanes.
  bool certifiedWrap = accumulator.getUpdateOverflow() == ondrix::ondsp::OverflowMode::Wrap;
  if (!certifiedWrap)
    certifiedWrap = llvm::all_of(outputs, [](ConstantRowOutput &output) {
      return succeeded(
          ondrix::analysis::FixedPointPrefixRangePlanner::proveOrderedZeroSeededConstantReduction(
              output.reduce, output.coefficients));
    });
  int64_t termGroup = 0;
  if (certifiedWrap) {
    termGroup = std::numeric_limits<int64_t>::max();
    for (ConstantRowOutput &output : outputs)
      termGroup = std::min(
          termGroup, ondrix::analysis::FixedPointPrefixRangePlanner::largestCertifiedTermGroup(
                         output.reduce, output.coefficients, /*termWidth=*/32));
  }
  int64_t groupedTerms = termGroup >= 2 ? (length / termGroup) * termGroup : 0;

  auto laneAccumulator = ondrix::ondsp::AccType::get(
      context, accumulator.getStorage(), accumulator.getFrac(), accumulator.getSignedness(),
      certifiedWrap ? ondrix::ondsp::OverflowMode::Wrap : accumulator.getUpdateOverflow(),
      static_cast<unsigned>(lanes));
  ondrix::ondsp::AccType groupAccumulator;
  ondrix::ondsp::FixedAttr groupTermNumeric;
  VectorType groupTermType;
  if (groupedTerms > 0) {
    groupAccumulator = ondrix::ondsp::AccType::get(
        context, builder.getI32Type(), accumulator.getFrac(), accumulator.getSignedness(),
        ondrix::ondsp::OverflowMode::Wrap, static_cast<unsigned>(lanes));
    groupTermNumeric = ondrix::ondsp::FixedAttr::get(context, accumulator.getSignedness(),
                                                     builder.getI32Type(), accumulator.getFrac());
    groupTermType = VectorType::get({lanes}, builder.getI32Type());
  }
  auto columnType = VectorType::get({lanes}, storage);
  auto sampleType =
      VectorType::get({lanes}, cast<IntegerType>(first.exportOp.getDst().getStorage()));

  builder.setInsertionPoint(runEnd);
  // Above the budget the columns live in memory and the terms become a loop:
  // the block's instruction memory stops growing with the table, at the price
  // of the index arithmetic and the load the immediates did not need.
  bool loopForm = maxStraightLineCoefficients > 0 && length * lanes > maxStraightLineCoefficients &&
                  (groupedTerms == length || groupedTerms == 0);
  if (loopForm) {
    int64_t stride = groupedTerms == length ? termGroup : 1;
    Value table = createColumnTable(builder, loc, outputs, length, storage);
    Value zero = builder.create<arith::ConstantIndexOp>(loc, 0);
    Value one = builder.create<arith::ConstantIndexOp>(loc, 1);
    Value trips = builder.create<arith::ConstantIndexOp>(loc, length / stride);
    Value seed = builder.create<ondrix::ondsp::AccZeroOp>(loc, laneAccumulator);
    auto loop = builder.create<scf::ForOp>(
        loc, zero, trips, one, ValueRange{seed},
        [&](OpBuilder &body, Location bodyLoc, Value trip, ValueRange carried) {
          Value running = carried.front();
          Value inner = groupedTerms == length
                            ? body.create<ondrix::ondsp::AccZeroOp>(bodyLoc, groupAccumulator)
                            : running;
          Value base = stride == 1 ? trip
                                   : body.create<arith::MulIOp>(
                                         bodyLoc, trip,
                                         body.create<arith::ConstantIndexOp>(bodyLoc, stride));
          for (int64_t step = 0; step < stride; ++step) {
            Value index =
                step == 0 ? base
                          : body.create<arith::AddIOp>(
                                bodyLoc, base, body.create<arith::ConstantIndexOp>(bodyLoc, step));
            Value offset = body.create<arith::MulIOp>(
                bodyLoc, index, body.create<arith::ConstantIndexOp>(bodyLoc, lanes));
            Value column =
                body.create<vector::LoadOp>(bodyLoc, columnType, table, ValueRange{offset});
            Value element = body.create<memref::LoadOp>(bodyLoc, input, ValueRange{index});
            inner = body.create<ondrix::ondsp::MacOp>(
                bodyLoc, groupedTerms == length ? groupAccumulator : laneAccumulator, inner, column,
                element, numeric, *first.reduce.getProduct());
          }
          if (groupedTerms == length) {
            Value termValue = body.create<ondrix::ondsp::AccExportOp>(
                bodyLoc, groupTermType, inner, groupTermNumeric,
                ondrix::ondsp::RoundingMode::TowardNegative, ondrix::ondsp::OverflowMode::Wrap);
            inner = body.create<ondrix::ondsp::AccAddTermOp>(bodyLoc, laneAccumulator, running,
                                                             termValue, groupTermNumeric);
          }
          body.create<scf::YieldOp>(bodyLoc, inner);
        });
    emitBlockTail(builder, loc, loop.getResult(0), first, sampleType, lanes);
    eraseBatchedOutputs(outputs);
    return;
  }
  Value acc = builder.create<ondrix::ondsp::AccZeroOp>(loc, laneAccumulator);
  Value group;
  for (int64_t term = 0; term < length; ++term) {
    bool grouped = term < groupedTerms;
    if (grouped && term % termGroup == 0)
      group = builder.create<ondrix::ondsp::AccZeroOp>(loc, groupAccumulator);
    // Lane l carries row l's coefficient for this term: the table column.
    SmallVector<llvm::APInt> column;
    for (const ConstantRowOutput &output : outputs)
      column.push_back(output.coefficients[term]);
    Value coefficients =
        builder.create<arith::ConstantOp>(loc, DenseIntElementsAttr::get(columnType, column));
    Value index = builder.create<arith::ConstantIndexOp>(loc, term);
    Value element = builder.create<memref::LoadOp>(loc, input, ValueRange{index});
    if (!grouped) {
      acc = builder.create<ondrix::ondsp::MacOp>(loc, laneAccumulator, acc, coefficients, element,
                                                 numeric, *first.reduce.getProduct());
      continue;
    }
    group = builder.create<ondrix::ondsp::MacOp>(loc, groupAccumulator, group, coefficients,
                                                 element, numeric, *first.reduce.getProduct());
    if (term % termGroup == termGroup - 1) {
      Value termValue = builder.create<ondrix::ondsp::AccExportOp>(
          loc, groupTermType, group, groupTermNumeric, ondrix::ondsp::RoundingMode::TowardNegative,
          ondrix::ondsp::OverflowMode::Wrap);
      acc = builder.create<ondrix::ondsp::AccAddTermOp>(loc, laneAccumulator, acc, termValue,
                                                        groupTermNumeric);
    }
  }
  emitBlockTail(builder, loc, acc, first, sampleType, lanes);
  eraseBatchedOutputs(outputs);
}

/// Groups the unrolled constant-row outputs of one block into runs of
/// `vectorWidth` consecutive positions under one contract, and batches each.
void batchConstantRowOutputsInBlock(Block &block, int64_t vectorWidth, bool requantizedProducts,
                                    int64_t maxStraightLineCoefficients, OpBuilder &builder) {
  SmallVector<ConstantRowOutput> outputs;
  for (auto reduce : block.getOps<ondrix::ondsp::ReduceMacOp>())
    if (std::optional<ConstantRowOutput> output = matchConstantRowOutput(reduce))
      if (requantizedProducts || output->reduce.getProduct()->getShift() == 0)
        outputs.push_back(std::move(*output));
  if (static_cast<int64_t>(outputs.size()) < vectorWidth)
    return;
  llvm::stable_sort(outputs, [](const ConstantRowOutput &lhs, const ConstantRowOutput &rhs) {
    return lhs.position < rhs.position;
  });
  size_t start = 0;
  while (start + vectorWidth <= outputs.size()) {
    bool run = true;
    for (int64_t lane = 1; lane < vectorWidth && run; ++lane)
      run = outputs[start + lane].position == outputs[start].position + lane &&
            haveSameLaneContract(outputs[start], outputs[start + lane]);
    if (!run) {
      ++start;
      continue;
    }
    batchConstantRowOutputs(MutableArrayRef<ConstantRowOutput>(outputs).slice(start, vectorWidth),
                            builder, maxStraightLineCoefficients);
    start += vectorWidth;
  }
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
    if (chunkMultiple < 1 || vectorWidth * chunkMultiple > kMaxVectorWidth) {
      getOperation().emitError("chunk-multiple must be positive and keep the block within ")
          << kMaxVectorWidth << " lanes";
      signalPassFailure();
      return;
    }

    // Collect first: the batched loop this pass creates must never be offered
    // to the matcher, and the ordered loop is mutated in place.
    SmallVector<scf::ForOp> candidates;
    getOperation().walk([&](scf::ForOp loop) { candidates.push_back(loop); });

    // A sliding-window block may span chunk-multiple machine vectors of
    // outputs, stepping down one vector at a time, then by halves below one
    // vector: the widest block the output count fills, as the reduce ladders.
    SmallVector<int64_t> ladder;
    for (int64_t multiple = chunkMultiple; multiple >= 2; --multiple)
      ladder.push_back(vectorWidth * multiple);
    for (int64_t lanes = vectorWidth; lanes > 1; lanes /= 2)
      ladder.push_back(lanes);
    OpBuilder builder(&getContext());
    for (scf::ForOp loop : candidates) {
      bool batched = false;
      for (int64_t lanes : ladder) {
        if (FailureOr<DecimateLoopShape> shape = matchDecimateLoop(loop, lanes); succeeded(shape)) {
          if (!requantizedProducts && shape->product.getShift() != 0)
            break;
          batchDecimateOutputs(*shape, lanes, builder);
          batched = true;
          break;
        }
      }
      if (batched)
        continue;
      if (FailureOr<ColumnLoopShape> shape = matchColumnLoop(loop); succeeded(shape))
        if (requantizedProducts || shape->product.getShift() == 0)
          batchColumnOutputs(*shape, vectorWidth, builder);
    }

    SmallVector<Block *> blocks;
    getOperation().walk([&](func::FuncOp function) {
      for (Block &block : function.getBody())
        blocks.push_back(&block);
    });
    for (Block *block : blocks)
      for (int64_t lanes = vectorWidth; lanes > 1; lanes /= 2)
        batchConstantRowOutputsInBlock(*block, lanes, requantizedProducts,
                                       maxStraightLineCoefficients, builder);
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
