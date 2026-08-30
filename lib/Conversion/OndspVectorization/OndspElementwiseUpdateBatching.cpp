#include "ondrix/Conversion/OndspVectorization/OndspVectorization.h"
#include "ondrix/Conversion/Utils/MemRefLayoutUtils.h"

#include "ondrix/Dialect/ondsp/IR/OndspDialect.h"
#include "ondrix/Dialect/ondsp/IR/OndspOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"

#include <optional>

namespace ondrix {
#define GEN_PASS_DEF_VECTORIZEONDSPFIXEDELEMENTWISEUPDATES
#include "ondrix/Conversion/Passes.h.inc"
} // namespace ondrix

using namespace mlir;

namespace {

/// Largest accepted batch width, bounding every index the rewrite derives.
constexpr int64_t kMaxVectorWidth = 4096;

/// Operation count of the matched body. The rewrite claims every one of them,
/// so an equal count plus pairwise distinct matches is an exact cover: nothing
/// unrecognized, and therefore no unmodelled side effect or second consumer,
/// can be hiding in the loop.
constexpr size_t kUpdateBodyOperations = 11;

/// Everything the matcher recovered from one bufferized elementwise update
/// loop. The loop is rewritten only when every field is present and
/// consistent, so a partially understood loop is never left behind.
struct ElementwiseUpdateLoopShape {
  scf::ForOp loop;
  /// Number of state elements the ordered loop updates.
  int64_t updateCount = 0;
  Value samples;
  Value state;
  /// The loop-invariant index the sample walk counts down from: index `k`
  /// reads `samples[base - k]`.
  Value sampleBase;
  /// The loop-invariant scalar factor of the declared product.
  Value step;
  /// Operand roles of the two commutative operations, recorded rather than
  /// normalized so the batched arithmetic is the matched arithmetic.
  bool stepIsProductLhs = false;
  bool stateIsSumLhs = false;
  /// The declared rounding boundary and the declared update overflow policy,
  /// both reused verbatim per lane.
  ondrix::ondsp::ScaleAttr scale;
  Attribute updateNumeric;
  IntegerType sampleElement;
  IntegerType productElement;
  IntegerType scaledElement;
  IntegerType stateElement;
  IntegerType sumElement;
};

/// Rank-1 memref with a static extent, a contiguous layout, and a memory space
/// the Vector to LLVM lowering accepts.
bool isBatchableRankOneMemRef(Value value) {
  auto type = dyn_cast<MemRefType>(value.getType());
  return type && type.getRank() == 1 && !type.isDynamicDim(0) && isLastMemrefDimUnitStride(type) &&
         ondrix::conversion::hasDefaultLLVMVectorMemorySpace(type);
}

/// Whether `value` is available where the loop is, rather than produced inside
/// it.
bool isAvailableAtLoop(scf::ForOp loop, Value value) {
  return !loop.getRegion().isAncestor(value.getParentRegion());
}

/// The producer of `value` when it is an `OpTy` in `body` and `value` has no
/// second consumer the rewrite would leave unserved.
template <typename OpTy> OpTy getSingleUseProducerIn(Value value, Block &body) {
  auto producer = value.getDefiningOp<OpTy>();
  if (!producer || producer->getBlock() != &body || !value.hasOneUse())
    return nullptr;
  return producer;
}

/// Matches the loop shape a bufferized elementwise fixed-point state update
/// emits: per index, one backward sample read, one product with a
/// loop-invariant step, one declared rounding boundary, one saturating add
/// into the state element the same index loads, and one store back to it.
/// Anything else fails closed and keeps the ordered schedule.
FailureOr<ElementwiseUpdateLoopShape> matchUpdateLoop(scf::ForOp loop, int64_t vectorWidth) {
  if (!loop.getInitArgs().empty())
    return failure();

  std::optional<int64_t> lowerBound = getConstantIntValue(loop.getLowerBound());
  std::optional<int64_t> upperBound = getConstantIntValue(loop.getUpperBound());
  std::optional<int64_t> step = getConstantIntValue(loop.getStep());
  if (!lowerBound || !upperBound || !step || *lowerBound != 0 || *step != 1 || *upperBound <= 0)
    return failure();

  Block &body = *loop.getBody();
  Value index = loop.getInductionVar();

  SmallVector<Operation *> operations;
  for (Operation &operation : body.without_terminator())
    operations.push_back(&operation);
  if (operations.size() != kUpdateBodyOperations)
    return failure();

  auto store = dyn_cast<memref::StoreOp>(operations.back());
  if (!store || store.getIndices().size() != 1 || store.getIndices().front() != index)
    return failure();

  auto updated = getSingleUseProducerIn<ondrix::ondsp::SatCastOp>(store.getValueToStore(), body);
  if (!updated)
    return failure();
  auto sum = getSingleUseProducerIn<arith::AddIOp>(updated.getInput(), body);
  if (!sum)
    return failure();
  auto lhsExtend = getSingleUseProducerIn<arith::ExtSIOp>(sum.getLhs(), body);
  auto rhsExtend = getSingleUseProducerIn<arith::ExtSIOp>(sum.getRhs(), body);
  if (!lhsExtend || !rhsExtend)
    return failure();

  // The state element and the rounded product reach the add in either operand
  // order; the memref load is what tells them apart.
  bool stateIsSumLhs = getSingleUseProducerIn<memref::LoadOp>(lhsExtend.getIn(), body) != nullptr;
  arith::ExtSIOp stateExtend = stateIsSumLhs ? lhsExtend : rhsExtend;
  arith::ExtSIOp scaledExtend = stateIsSumLhs ? rhsExtend : lhsExtend;

  auto stateLoad = getSingleUseProducerIn<memref::LoadOp>(stateExtend.getIn(), body);
  auto scaled = getSingleUseProducerIn<ondrix::ondsp::RoundShiftOp>(scaledExtend.getIn(), body);
  if (!stateLoad || !scaled)
    return failure();
  if (stateLoad.getMemRef() != store.getMemRef() || stateLoad.getIndices().size() != 1 ||
      stateLoad.getIndices().front() != index)
    return failure();

  auto product = getSingleUseProducerIn<arith::MulIOp>(scaled.getInput(), body);
  if (!product)
    return failure();
  bool stepIsProductLhs = isAvailableAtLoop(loop, product.getLhs());
  Value stepValue = stepIsProductLhs ? product.getLhs() : product.getRhs();
  Value sampleWide = stepIsProductLhs ? product.getRhs() : product.getLhs();
  if (!isAvailableAtLoop(loop, stepValue))
    return failure();

  auto sampleExtend = getSingleUseProducerIn<arith::ExtSIOp>(sampleWide, body);
  if (!sampleExtend)
    return failure();
  auto sampleLoad = getSingleUseProducerIn<memref::LoadOp>(sampleExtend.getIn(), body);
  if (!sampleLoad || sampleLoad.getIndices().size() != 1)
    return failure();
  auto walk = getSingleUseProducerIn<arith::SubIOp>(sampleLoad.getIndices().front(), body);
  if (!walk || walk.getRhs() != index || !isAvailableAtLoop(loop, walk.getLhs()))
    return failure();

  SmallPtrSet<Operation *, kUpdateBodyOperations> matched{
      walk,        sampleLoad,   sampleExtend, product, scaled, stateLoad,
      stateExtend, scaledExtend, sum,          updated, store};
  if (matched.size() != kUpdateBodyOperations)
    return failure();

  ElementwiseUpdateLoopShape shape;
  shape.loop = loop;
  shape.updateCount = *upperBound;
  shape.samples = sampleLoad.getMemRef();
  shape.state = store.getMemRef();
  shape.sampleBase = walk.getLhs();
  shape.step = stepValue;
  shape.stepIsProductLhs = stepIsProductLhs;
  shape.stateIsSumLhs = stateIsSumLhs;
  shape.scale = scaled.getScale();
  shape.updateNumeric = updated.getNumeric();

  if (!isBatchableRankOneMemRef(shape.samples) || !isBatchableRankOneMemRef(shape.state))
    return failure();

  shape.sampleElement =
      dyn_cast<IntegerType>(cast<MemRefType>(shape.samples.getType()).getElementType());
  shape.stateElement =
      dyn_cast<IntegerType>(cast<MemRefType>(shape.state.getType()).getElementType());
  shape.productElement = dyn_cast<IntegerType>(product.getType());
  shape.scaledElement = dyn_cast<IntegerType>(scaled.getResult().getType());
  shape.sumElement = dyn_cast<IntegerType>(sum.getType());
  if (!shape.sampleElement || !shape.stateElement || !shape.productElement ||
      !shape.scaledElement || !shape.sumElement)
    return failure();

  // The rewrite defers a block's stores past all of that block's sample loads,
  // so it is only sound when the two sequences are distinct storage; the
  // refusal set and the residual precondition are in the pass description.
  if (ondrix::conversion::mayShareStorage(shape.samples, shape.state))
    return failure();

  int64_t fullBlocks = shape.updateCount / vectorWidth;
  if (fullBlocks < 1)
    return failure();

  // The batched span is the set the ordered block already read, so only the
  // widths have to be pinned: the state must hold every ordered index, and the
  // samples must be long enough for a contiguous width-sized load to exist.
  int64_t stateLength = cast<MemRefType>(shape.state.getType()).getDimSize(0);
  int64_t sampleLength = cast<MemRefType>(shape.samples.getType()).getDimSize(0);
  if (shape.updateCount > stateLength || vectorWidth > sampleLength)
    return failure();

  return shape;
}

/// Replaces the leading full blocks of an ordered elementwise update loop with
/// a batched loop over `vectorWidth` state elements at a time and moves the
/// ordered loop's lower bound past them. The ordered body is not touched, so
/// the remaining elements keep exactly the schedule they had.
void batchElementwiseUpdates(const ElementwiseUpdateLoopShape &shape, int64_t vectorWidth,
                             OpBuilder &builder) {
  scf::ForOp loop = shape.loop;
  Location loc = loop.getLoc();
  int64_t fullBlocks = shape.updateCount / vectorWidth;
  int64_t batchedUpdates = fullBlocks * vectorWidth;

  auto sampleLanes = VectorType::get({vectorWidth}, shape.sampleElement);
  auto productLanes = VectorType::get({vectorWidth}, shape.productElement);
  auto scaledLanes = VectorType::get({vectorWidth}, shape.scaledElement);
  auto stateLanes = VectorType::get({vectorWidth}, shape.stateElement);
  auto sumLanes = VectorType::get({vectorWidth}, shape.sumElement);

  // The span load runs forward and the ordered walk runs backward, so lane `i`
  // of the block starting at `j` must receive span element `W - 1 - i`.
  SmallVector<int64_t> reversedLanes;
  for (int64_t lane = 0; lane < vectorWidth; ++lane)
    reversedLanes.push_back(vectorWidth - 1 - lane);

  builder.setInsertionPoint(loop);
  Value zeroIndex = builder.create<arith::ConstantIndexOp>(loc, 0);
  Value lastLane = builder.create<arith::ConstantIndexOp>(loc, vectorWidth - 1);
  Value batchedEnd = builder.create<arith::ConstantIndexOp>(loc, batchedUpdates);
  Value batchStep = builder.create<arith::ConstantIndexOp>(loc, vectorWidth);
  Value stepLanes = builder.create<vector::SplatOp>(loc, productLanes, shape.step);

  builder.create<scf::ForOp>(
      loc, zeroIndex, batchedEnd, batchStep, ValueRange{},
      [&](OpBuilder &blockBuilder, Location blockLoc, Value blockStart, ValueRange) {
        Value blockTop = blockBuilder.create<arith::SubIOp>(blockLoc, shape.sampleBase, blockStart);
        Value spanBase = blockBuilder.create<arith::SubIOp>(blockLoc, blockTop, lastLane);
        Value span = blockBuilder.create<vector::LoadOp>(blockLoc, sampleLanes, shape.samples,
                                                         ValueRange{spanBase});
        Value samples = blockBuilder.create<vector::ShuffleOp>(blockLoc, span, span, reversedLanes);
        Value wide = blockBuilder.create<arith::ExtSIOp>(blockLoc, productLanes, samples);
        Value product =
            shape.stepIsProductLhs
                ? blockBuilder.create<arith::MulIOp>(blockLoc, stepLanes, wide).getResult()
                : blockBuilder.create<arith::MulIOp>(blockLoc, wide, stepLanes).getResult();
        Value scaled = blockBuilder.create<ondrix::ondsp::RoundShiftOp>(blockLoc, scaledLanes,
                                                                        product, shape.scale);
        Value state = blockBuilder.create<vector::LoadOp>(blockLoc, stateLanes, shape.state,
                                                          ValueRange{blockStart});
        Value stateWide = blockBuilder.create<arith::ExtSIOp>(blockLoc, sumLanes, state);
        Value scaledWide = blockBuilder.create<arith::ExtSIOp>(blockLoc, sumLanes, scaled);
        Value sum =
            shape.stateIsSumLhs
                ? blockBuilder.create<arith::AddIOp>(blockLoc, stateWide, scaledWide).getResult()
                : blockBuilder.create<arith::AddIOp>(blockLoc, scaledWide, stateWide).getResult();
        Value updated = blockBuilder.create<ondrix::ondsp::SatCastOp>(blockLoc, stateLanes, sum,
                                                                      shape.updateNumeric);
        blockBuilder.create<vector::StoreOp>(blockLoc, updated, shape.state,
                                             ValueRange{blockStart});
        blockBuilder.create<scf::YieldOp>(blockLoc);
      });

  // A fully covered ordered loop is erased rather than left dead, which is the
  // standing rule for every batcher here.
  if (batchedUpdates == shape.updateCount) {
    loop.erase();
    return;
  }
  loop.getLowerBoundMutable().assign(batchedEnd);
}

class VectorizeOndspFixedElementwiseUpdatesPass final
    : public ondrix::impl::VectorizeOndspFixedElementwiseUpdatesBase<
          VectorizeOndspFixedElementwiseUpdatesPass> {
public:
  using ondrix::impl::VectorizeOndspFixedElementwiseUpdatesBase<
      VectorizeOndspFixedElementwiseUpdatesPass>::VectorizeOndspFixedElementwiseUpdatesBase;

  void runOnOperation() override {
    if (vectorWidth <= 1) {
      getOperation().emitError("vector-width must be greater than one");
      signalPassFailure();
      return;
    }
    // The width becomes a static shuffle mask and every index the block
    // derives, so it is bounded from above as well.
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
      FailureOr<ElementwiseUpdateLoopShape> shape = matchUpdateLoop(loop, vectorWidth);
      if (failed(shape))
        continue;
      batchElementwiseUpdates(*shape, vectorWidth, builder);
    }
  }
};

} // namespace

std::unique_ptr<Pass> ondrix::createVectorizeOndspFixedElementwiseUpdatesPass() {
  return std::make_unique<VectorizeOndspFixedElementwiseUpdatesPass>();
}

std::unique_ptr<Pass> ondrix::createVectorizeOndspFixedElementwiseUpdatesPass(
    const VectorizeOndspFixedElementwiseUpdatesOptions &options) {
  return std::make_unique<VectorizeOndspFixedElementwiseUpdatesPass>(options);
}
