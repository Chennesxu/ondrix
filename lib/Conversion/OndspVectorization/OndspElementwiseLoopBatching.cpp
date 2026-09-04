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

#include "llvm/ADT/SmallVector.h"

#include <optional>

namespace ondrix {
#define GEN_PASS_DEF_VECTORIZEONDSPFIXEDELEMENTWISELOOPS
#include "ondrix/Conversion/Passes.h.inc"
} // namespace ondrix

using namespace mlir;

namespace {

/// Largest accepted batch width, bounding every index the rewrite derives.
constexpr int64_t kMaxVectorWidth = 4096;

/// Longest accepted elementwise chain between the load and the store. The
/// chain is re-emitted once per block, so an unbounded one would trade the
/// ordered loop's compact body for an arbitrarily large one.
constexpr size_t kMaxChainOperations = 8;

/// Everything the matcher recovered from one bufferized elementwise loop. The
/// loop is rewritten only when every field is present and consistent, so a
/// partially understood loop is never left behind.
struct ElementwiseLoopShape {
  scf::ForOp loop;
  /// Number of elements the ordered loop transforms.
  int64_t elementCount = 0;
  Value input;
  Value output;
  /// The chain between the load and the store, in emission order.
  SmallVector<Operation *> chain;
};

/// Rank-1 memref with a static extent, a contiguous layout, and a memory space
/// the Vector to LLVM lowering accepts.
bool isBatchableRankOneMemRef(Value value) {
  auto type = dyn_cast<MemRefType>(value.getType());
  return type && type.getRank() == 1 && !type.isDynamicDim(0) && isLastMemrefDimUnitStride(type) &&
         isa<IntegerType>(type.getElementType()) &&
         ondrix::conversion::hasDefaultLLVMVectorMemorySpace(type);
}

/// Whether `operation` is one of the elementwise integer transformations the
/// batched body can re-emit lane by lane.
bool isBatchableElementwiseOp(Operation *operation) {
  return isa<ondrix::ondsp::RoundShiftOp, arith::ExtSIOp, arith::TruncIOp>(operation);
}

/// Matches the loop shape a bufferized elementwise requantization emits: per
/// index, one load, a chain of elementwise integer transformations, and one
/// store at the same index into a different sequence. Anything else fails
/// closed and keeps the ordered schedule.
FailureOr<ElementwiseLoopShape> matchElementwiseLoop(scf::ForOp loop, int64_t vectorWidth) {
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
  if (operations.size() < 2 || operations.size() > kMaxChainOperations + 2)
    return failure();

  auto load = dyn_cast<memref::LoadOp>(operations.front());
  auto store = dyn_cast<memref::StoreOp>(operations.back());
  if (!load || !store || load.getIndices().size() != 1 || store.getIndices().size() != 1 ||
      load.getIndices().front() != index || store.getIndices().front() != index)
    return failure();

  // In-order walk, each operation consuming the previous value with no second
  // use: the operation count is then an exact cover, so no unrecognized side
  // effect or escaping value can hide in the loop.
  ElementwiseLoopShape shape;
  Value carried = load.getResult();
  for (Operation *operation : llvm::drop_begin(llvm::drop_end(operations))) {
    if (!isBatchableElementwiseOp(operation) || operation->getNumOperands() != 1 ||
        operation->getNumResults() != 1 || operation->getOperand(0) != carried ||
        !carried.hasOneUse() || !isa<IntegerType>(operation->getResult(0).getType()))
      return failure();
    shape.chain.push_back(operation);
    carried = operation->getResult(0);
  }
  if (store.getValueToStore() != carried || !carried.hasOneUse())
    return failure();

  shape.loop = loop;
  shape.elementCount = *upperBound;
  shape.input = load.getMemRef();
  shape.output = store.getMemRef();
  if (!isBatchableRankOneMemRef(shape.input) || !isBatchableRankOneMemRef(shape.output))
    return failure();

  // The rewrite defers a block's store past that block's loads, so it is only
  // sound when the two sequences are distinct storage; the refusal set and the
  // residual precondition are in the pass description.
  if (ondrix::conversion::mayShareStorage(shape.input, shape.output))
    return failure();

  if (shape.elementCount / vectorWidth < 1)
    return failure();
  int64_t inputLength = cast<MemRefType>(shape.input.getType()).getDimSize(0);
  int64_t outputLength = cast<MemRefType>(shape.output.getType()).getDimSize(0);
  if (shape.elementCount > inputLength || shape.elementCount > outputLength)
    return failure();

  return shape;
}

/// Replaces the leading full blocks of an ordered elementwise loop with a
/// batched loop over `vectorWidth` elements at a time and moves the ordered
/// loop's lower bound past them. The ordered body is not touched, so the
/// remaining elements keep exactly the schedule they had.
void batchElementwiseLoop(const ElementwiseLoopShape &shape, int64_t vectorWidth,
                          OpBuilder &builder) {
  scf::ForOp loop = shape.loop;
  Location loc = loop.getLoc();
  int64_t batchedElements = (shape.elementCount / vectorWidth) * vectorWidth;
  auto lanesOf = [&](Type element) { return VectorType::get({vectorWidth}, element); };

  builder.setInsertionPoint(loop);
  Value zeroIndex = builder.create<arith::ConstantIndexOp>(loc, 0);
  Value batchedEnd = builder.create<arith::ConstantIndexOp>(loc, batchedElements);
  Value batchStep = builder.create<arith::ConstantIndexOp>(loc, vectorWidth);
  auto inputLanes = lanesOf(cast<MemRefType>(shape.input.getType()).getElementType());

  builder.create<scf::ForOp>(
      loc, zeroIndex, batchedEnd, batchStep, ValueRange{},
      [&](OpBuilder &blockBuilder, Location blockLoc, Value blockStart, ValueRange) {
        Value value = blockBuilder.create<vector::LoadOp>(blockLoc, inputLanes, shape.input,
                                                          ValueRange{blockStart});
        for (Operation *operation : shape.chain) {
          Type lanes = lanesOf(operation->getResult(0).getType());
          if (auto roundShift = dyn_cast<ondrix::ondsp::RoundShiftOp>(operation)) {
            value = blockBuilder.create<ondrix::ondsp::RoundShiftOp>(blockLoc, lanes, value,
                                                                     roundShift.getScale());
          } else if (isa<arith::ExtSIOp>(operation)) {
            value = blockBuilder.create<arith::ExtSIOp>(blockLoc, lanes, value);
          } else {
            value = blockBuilder.create<arith::TruncIOp>(blockLoc, lanes, value);
          }
        }
        blockBuilder.create<vector::StoreOp>(blockLoc, value, shape.output, ValueRange{blockStart});
        blockBuilder.create<scf::YieldOp>(blockLoc);
      });

  // A fully covered ordered loop is erased rather than left dead, which is the
  // standing rule for every batcher here.
  if (batchedElements == shape.elementCount) {
    loop.erase();
    return;
  }
  loop.getLowerBoundMutable().assign(batchedEnd);
}

class VectorizeOndspFixedElementwiseLoopsPass final
    : public ondrix::impl::VectorizeOndspFixedElementwiseLoopsBase<
          VectorizeOndspFixedElementwiseLoopsPass> {
public:
  using ondrix::impl::VectorizeOndspFixedElementwiseLoopsBase<
      VectorizeOndspFixedElementwiseLoopsPass>::VectorizeOndspFixedElementwiseLoopsBase;

  void runOnOperation() override {
    if (vectorWidth <= 1) {
      getOperation().emitError("vector-width must be greater than one");
      signalPassFailure();
      return;
    }
    // The width becomes every index the block derives, so it is bounded from
    // above as well.
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
      FailureOr<ElementwiseLoopShape> shape = matchElementwiseLoop(loop, vectorWidth);
      if (failed(shape))
        continue;
      batchElementwiseLoop(*shape, vectorWidth, builder);
    }
  }
};

} // namespace

std::unique_ptr<Pass> ondrix::createVectorizeOndspFixedElementwiseLoopsPass() {
  return std::make_unique<VectorizeOndspFixedElementwiseLoopsPass>();
}

std::unique_ptr<Pass> ondrix::createVectorizeOndspFixedElementwiseLoopsPass(
    const VectorizeOndspFixedElementwiseLoopsOptions &options) {
  return std::make_unique<VectorizeOndspFixedElementwiseLoopsPass>(options);
}
