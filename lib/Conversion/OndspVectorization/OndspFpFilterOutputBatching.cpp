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

  builder.create<scf::ForOp>(
      loc, zeroIndex, batchedEnd, batchStep, ValueRange{},
      [&](OpBuilder &blockBuilder, Location blockLoc, Value blockStart, ValueRange) {
        // The taps stay a loop: an f32 vector accumulator is a native
        // loop-carried value, unlike the multi-lane fixed accumulator the
        // decimate batching must unroll around.
        auto taps = blockBuilder.create<scf::ForOp>(
            blockLoc, zeroIndex, tapEnd, oneIndex, ValueRange{laneZero},
            [&](OpBuilder &tapBuilder, Location tapLoc, Value tap, ValueRange iterArgs) {
              Value base = tapBuilder.create<arith::AddIOp>(tapLoc, blockStart, tap);
              Value values = tapBuilder.create<vector::LoadOp>(tapLoc, laneType, shape.input,
                                                               ValueRange{base});
              Value coefficient =
                  tapBuilder.create<memref::LoadOp>(tapLoc, shape.coefficients, ValueRange{tap});
              Value splat = tapBuilder.create<vector::SplatOp>(tapLoc, laneType, coefficient);
              Value updated;
              if (shape.contract == ondrix::ondsp::FpContractMode::Fma) {
                // One fused event per lane per tap, exactly the scalar chain.
                updated = tapBuilder.create<math::FmaOp>(tapLoc, values, splat, iterArgs.front());
              } else if (shape.contract == ondrix::ondsp::FpContractMode::Fast && fuseFast) {
                // fast admits both members; selecting the fused one spends F.
                // The batch itself is order-preserving, so R is never spent.
                updated = ondrix::ondsp::consumeFastPermission(
                    tapBuilder.create<math::FmaOp>(tapLoc, values, splat, iterArgs.front()),
                    ondrix::ondsp::FastPermission::FuseMultiplyAdd);
              } else {
                // Separate ordered product and accumulation events per lane,
                // in the scalar chain's operand order.
                Value product = tapBuilder.create<arith::MulFOp>(tapLoc, values, splat);
                updated = tapBuilder.create<arith::AddFOp>(tapLoc, iterArgs.front(), product);
              }
              tapBuilder.create<scf::YieldOp>(tapLoc, updated);
            });
        blockBuilder.create<vector::StoreOp>(blockLoc, taps.getResult(0), shape.output,
                                             ValueRange{blockStart});
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
      FailureOr<FpFilterLoopShape> shape = matchFpFilterLoop(loop, vectorWidth);
      if (succeeded(shape))
        batchFpFilterOutputs(*shape, vectorWidth, supportsVectorFma, builder);
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
