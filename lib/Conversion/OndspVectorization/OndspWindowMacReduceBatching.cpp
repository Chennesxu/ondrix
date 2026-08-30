#include "ondrix/Conversion/OndspVectorization/OndspVectorization.h"
#include "ondrix/Conversion/Utils/MemRefLayoutUtils.h"

#include "ondrix/Dialect/ondsp/IR/OndspDialect.h"
#include "ondrix/Dialect/ondsp/IR/OndspOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"

namespace ondrix {
#define GEN_PASS_DEF_VECTORIZEONDSPFIXEDWINDOWMACREDUCE
#include "ondrix/Conversion/Passes.h.inc"
} // namespace ondrix

using namespace mlir;

namespace {

/// Largest accepted batch width, bounding every index the rewrite derives.
constexpr int64_t kMaxVectorWidth = 4096;

/// Carrier the batched product narrows to when the declared one is wider than
/// the exact product needs.
constexpr unsigned kNarrowProductWidth = 32;

/// Operation count of the matched body. The rewrite claims every one of them,
/// so an equal count plus pairwise distinct matches is an exact cover.
constexpr size_t kReduceBodyOperations = 7;

/// Everything the matcher recovered from one backward-window MAC reduction.
struct WindowMacReduceShape {
  scf::ForOp loop;
  /// Number of terms the ordered loop folds.
  int64_t termCount = 0;
  Value samples;
  Value coefficients;
  /// The loop-invariant index the sample walk counts down from: term `k`
  /// reads `samples[base - k]`.
  Value sampleBase;
  /// Operand roles of the two commutative operations, recorded rather than
  /// normalized so the batched arithmetic is the matched arithmetic.
  bool sampleIsProductLhs = false;
  bool carriedIsSumLhs = false;
  IntegerType sampleElement;
  IntegerType coefficientElement;
  IntegerType accumulatorElement;
  /// Carrier the batched product runs in; the declared one unless narrowed.
  IntegerType productElement;
};

/// Rank-1 memref with a static extent, a contiguous layout, and a memory space
/// the Vector to LLVM lowering accepts.
bool isBatchableRankOneMemRef(Value value) {
  auto type = dyn_cast<MemRefType>(value.getType());
  return type && type.getRank() == 1 && !type.isDynamicDim(0) && isLastMemrefDimUnitStride(type) &&
         ondrix::conversion::hasDefaultLLVMVectorMemorySpace(type);
}

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

/// Matches the loop a convolution-shaped fixed reduction lowers to: per term,
/// one backward sample read, one forward coefficient read, one product, and
/// one wrapping accumulate. Anything else fails closed.
FailureOr<WindowMacReduceShape> matchWindowMacReduce(scf::ForOp loop, int64_t vectorWidth) {
  if (loop.getNumRegionIterArgs() != 1)
    return failure();

  std::optional<int64_t> lowerBound = getConstantIntValue(loop.getLowerBound());
  std::optional<int64_t> upperBound = getConstantIntValue(loop.getUpperBound());
  std::optional<int64_t> step = getConstantIntValue(loop.getStep());
  if (!lowerBound || !upperBound || !step || *lowerBound != 0 || *step != 1 || *upperBound <= 0)
    return failure();

  Block &body = *loop.getBody();
  Value index = loop.getInductionVar();
  Value carried = loop.getRegionIterArgs().front();
  auto accumulator = dyn_cast<IntegerType>(carried.getType());
  if (!accumulator)
    return failure();

  SmallVector<Operation *> operations;
  for (Operation &operation : body.without_terminator())
    operations.push_back(&operation);
  if (operations.size() != kReduceBodyOperations)
    return failure();

  auto yield = cast<scf::YieldOp>(body.getTerminator());
  auto sum = getSingleUseProducerIn<arith::AddIOp>(yield.getOperand(0), body);
  if (!sum || sum.getType() != accumulator)
    return failure();

  // The carried value and the fresh product reach the add in either operand
  // order; only the carried one is the loop's own region argument.
  bool carriedIsSumLhs = sum.getLhs() == carried;
  if (!carriedIsSumLhs && sum.getRhs() != carried)
    return failure();
  Value productValue = carriedIsSumLhs ? sum.getRhs() : sum.getLhs();

  auto product = getSingleUseProducerIn<arith::MulIOp>(productValue, body);
  if (!product)
    return failure();
  auto lhsExtend = getSingleUseProducerIn<arith::ExtSIOp>(product.getLhs(), body);
  auto rhsExtend = getSingleUseProducerIn<arith::ExtSIOp>(product.getRhs(), body);
  if (!lhsExtend || !rhsExtend)
    return failure();

  auto lhsLoad = getSingleUseProducerIn<memref::LoadOp>(lhsExtend.getIn(), body);
  auto rhsLoad = getSingleUseProducerIn<memref::LoadOp>(rhsExtend.getIn(), body);
  if (!lhsLoad || !rhsLoad || lhsLoad.getIndices().size() != 1 || rhsLoad.getIndices().size() != 1)
    return failure();

  // The coefficient walk is the induction variable itself; the sample walk
  // counts down from a loop-invariant base. That is what tells them apart.
  bool sampleIsProductLhs = lhsLoad.getIndices().front() != index;
  memref::LoadOp sampleLoad = sampleIsProductLhs ? lhsLoad : rhsLoad;
  memref::LoadOp coefficientLoad = sampleIsProductLhs ? rhsLoad : lhsLoad;
  if (coefficientLoad.getIndices().front() != index)
    return failure();

  auto walk = getSingleUseProducerIn<arith::SubIOp>(sampleLoad.getIndices().front(), body);
  if (!walk || walk.getRhs() != index || !isAvailableAtLoop(loop, walk.getLhs()))
    return failure();

  SmallPtrSet<Operation *, kReduceBodyOperations> matched{
      walk, sampleLoad, coefficientLoad, lhsExtend, rhsExtend, product, sum};
  if (matched.size() != kReduceBodyOperations)
    return failure();

  WindowMacReduceShape shape;
  shape.loop = loop;
  shape.termCount = *upperBound;
  shape.samples = sampleLoad.getMemRef();
  shape.coefficients = coefficientLoad.getMemRef();
  shape.sampleBase = walk.getLhs();
  shape.sampleIsProductLhs = sampleIsProductLhs;
  shape.carriedIsSumLhs = carriedIsSumLhs;
  shape.accumulatorElement = accumulator;

  if (!isBatchableRankOneMemRef(shape.samples) || !isBatchableRankOneMemRef(shape.coefficients))
    return failure();
  shape.sampleElement =
      dyn_cast<IntegerType>(cast<MemRefType>(shape.samples.getType()).getElementType());
  shape.coefficientElement =
      dyn_cast<IntegerType>(cast<MemRefType>(shape.coefficients.getType()).getElementType());
  if (!shape.sampleElement || !shape.coefficientElement)
    return failure();

  // The one range obligation. Reassociating the accumulate needs no proof at
  // all — two's-complement addition is associative and commutative, so any
  // fold order of the same terms yields the same wrapped value — but a product
  // carrier only holds the exact product when it has both factors' widths.
  uint64_t exactProductWidth =
      uint64_t{shape.sampleElement.getWidth()} + shape.coefficientElement.getWidth();
  if (exactProductWidth <= kNarrowProductWidth)
    shape.productElement = IntegerType::get(loop.getContext(), kNarrowProductWidth);
  else if (exactProductWidth <= accumulator.getWidth())
    shape.productElement = accumulator;
  else
    return failure();

  int64_t fullBlocks = shape.termCount / vectorWidth;
  if (fullBlocks < 1)
    return failure();

  // The batched span is the set the ordered block already read, so only the
  // widths have to be pinned.
  int64_t coefficientLength = cast<MemRefType>(shape.coefficients.getType()).getDimSize(0);
  int64_t sampleLength = cast<MemRefType>(shape.samples.getType()).getDimSize(0);
  if (shape.termCount > coefficientLength || vectorWidth > sampleLength)
    return failure();

  return shape;
}

/// Folds the leading full blocks of a backward-window MAC reduction across
/// vector lanes and seeds the ordered remainder with their sum.
void batchWindowMacReduce(const WindowMacReduceShape &shape, int64_t vectorWidth,
                          OpBuilder &builder) {
  scf::ForOp loop = shape.loop;
  Location loc = loop.getLoc();
  int64_t fullBlocks = shape.termCount / vectorWidth;
  int64_t batchedTerms = fullBlocks * vectorWidth;

  auto sampleLanes = VectorType::get({vectorWidth}, shape.sampleElement);
  auto coefficientLanes = VectorType::get({vectorWidth}, shape.coefficientElement);
  auto productLanes = VectorType::get({vectorWidth}, shape.productElement);
  auto accumulatorLanes = VectorType::get({vectorWidth}, shape.accumulatorElement);

  // The span load runs forward and the ordered walk runs backward, so lane `i`
  // of the block starting at `j` must receive span element `W - 1 - i`.
  SmallVector<int64_t> reversedLanes;
  for (int64_t lane = 0; lane < vectorWidth; ++lane)
    reversedLanes.push_back(vectorWidth - 1 - lane);

  builder.setInsertionPoint(loop);
  Value zeroIndex = builder.create<arith::ConstantIndexOp>(loc, 0);
  Value lastLane = builder.create<arith::ConstantIndexOp>(loc, vectorWidth - 1);
  Value batchedEnd = builder.create<arith::ConstantIndexOp>(loc, batchedTerms);
  Value batchStep = builder.create<arith::ConstantIndexOp>(loc, vectorWidth);
  Value zeroLanes = builder.create<arith::ConstantOp>(
      loc, accumulatorLanes,
      DenseElementsAttr::get(accumulatorLanes,
                             APInt::getZero(shape.accumulatorElement.getWidth())));

  auto blockLoop = builder.create<scf::ForOp>(
      loc, zeroIndex, batchedEnd, batchStep, ValueRange{zeroLanes},
      [&](OpBuilder &blockBuilder, Location blockLoc, Value blockStart, ValueRange blockArgs) {
        Value blockTop = blockBuilder.create<arith::SubIOp>(blockLoc, shape.sampleBase, blockStart);
        Value spanBase = blockBuilder.create<arith::SubIOp>(blockLoc, blockTop, lastLane);
        Value span = blockBuilder.create<vector::LoadOp>(blockLoc, sampleLanes, shape.samples,
                                                         ValueRange{spanBase});
        Value samples = blockBuilder.create<vector::ShuffleOp>(blockLoc, span, span, reversedLanes);
        Value coefficients = blockBuilder.create<vector::LoadOp>(
            blockLoc, coefficientLanes, shape.coefficients, ValueRange{blockStart});
        Value sampleWide = blockBuilder.create<arith::ExtSIOp>(blockLoc, productLanes, samples);
        Value coefficientWide =
            blockBuilder.create<arith::ExtSIOp>(blockLoc, productLanes, coefficients);
        Value product =
            shape.sampleIsProductLhs
                ? blockBuilder.create<arith::MulIOp>(blockLoc, sampleWide, coefficientWide)
                      .getResult()
                : blockBuilder.create<arith::MulIOp>(blockLoc, coefficientWide, sampleWide)
                      .getResult();
        Value widened = product;
        if (shape.productElement != shape.accumulatorElement)
          widened = blockBuilder.create<arith::ExtSIOp>(blockLoc, accumulatorLanes, product);
        Value next = shape.carriedIsSumLhs
                         ? blockBuilder.create<arith::AddIOp>(blockLoc, blockArgs.front(), widened)
                               .getResult()
                         : blockBuilder.create<arith::AddIOp>(blockLoc, widened, blockArgs.front())
                               .getResult();
        blockBuilder.create<scf::YieldOp>(blockLoc, next);
      });

  Value folded =
      builder.create<vector::ReductionOp>(loc, vector::CombiningKind::ADD, blockLoop.getResult(0));
  Value initial = loop.getInitArgs().front();
  Value seeded = shape.carriedIsSumLhs
                     ? builder.create<arith::AddIOp>(loc, initial, folded).getResult()
                     : builder.create<arith::AddIOp>(loc, folded, initial).getResult();

  // A fully covered ordered loop is erased rather than left dead, which is the
  // standing rule for every batcher here.
  if (batchedTerms == shape.termCount) {
    loop.getResult(0).replaceAllUsesWith(seeded);
    loop.erase();
    return;
  }
  loop.getInitArgsMutable().assign(seeded);
  loop.getLowerBoundMutable().assign(batchedEnd);
}

class VectorizeOndspFixedWindowMacReducePass final
    : public ondrix::impl::VectorizeOndspFixedWindowMacReduceBase<
          VectorizeOndspFixedWindowMacReducePass> {
public:
  using ondrix::impl::VectorizeOndspFixedWindowMacReduceBase<
      VectorizeOndspFixedWindowMacReducePass>::VectorizeOndspFixedWindowMacReduceBase;

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

    // Collect first: the block loop this pass creates must never be offered to
    // the matcher, and the ordered loop is mutated in place.
    SmallVector<scf::ForOp> candidates;
    getOperation().walk([&](scf::ForOp loop) { candidates.push_back(loop); });

    OpBuilder builder(&getContext());
    for (scf::ForOp loop : candidates) {
      FailureOr<WindowMacReduceShape> shape = matchWindowMacReduce(loop, vectorWidth);
      if (failed(shape))
        continue;
      batchWindowMacReduce(*shape, vectorWidth, builder);
    }
  }
};

} // namespace

std::unique_ptr<Pass> ondrix::createVectorizeOndspFixedWindowMacReducePass() {
  return std::make_unique<VectorizeOndspFixedWindowMacReducePass>();
}

std::unique_ptr<Pass> ondrix::createVectorizeOndspFixedWindowMacReducePass(
    const VectorizeOndspFixedWindowMacReduceOptions &options) {
  return std::make_unique<VectorizeOndspFixedWindowMacReducePass>(options);
}
