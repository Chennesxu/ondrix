#include "ondrix/Transforms/Passes.h"

#include "ondrix/Dialect/ondsp/IR/OndspDialect.h"
#include "ondrix/Dialect/ondsp/IR/OndspOps.h"
#include "ondrix/Dialect/ondsp/IR/OndspSemantics.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"

#include "llvm/Support/MathExtras.h"

namespace ondrix {
#define GEN_PASS_DEF_UNROLLONDSPFIXEDMACLOOPS
#include "ondrix/Transforms/Passes.h.inc"
} // namespace ondrix

using namespace mlir;
using namespace ondrix::ondsp;

namespace {

/// Trip-count bound below which the chain goes straight-line; the measured
/// argument is the pass description's.
constexpr int64_t kMaxUnrolledTrip = 64;

/// A compile-time index value, seen either directly or through `tensor.dim`
/// of a statically shaped tensor.
std::optional<int64_t> getStaticIndex(Value value) {
  if (std::optional<int64_t> constant = getConstantIntValue(value))
    return constant;
  auto dim = value.getDefiningOp<tensor::DimOp>();
  if (!dim)
    return std::nullopt;
  auto type = dyn_cast<RankedTensorType>(dim.getSource().getType());
  std::optional<int64_t> index = dim.getConstantIndex();
  if (!type || !index || type.isDynamicDim(*index))
    return std::nullopt;
  return type.getDimSize(*index);
}

// Every intermediate here can overflow i64 for legal extreme bounds, and a
// wrapped trip count once deleted a one-iteration loop outright; anything
// that overflows is refused instead.
std::optional<int64_t> getUnrollableTripCount(scf::ForOp loop) {
  std::optional<int64_t> lower = getStaticIndex(loop.getLowerBound());
  std::optional<int64_t> upper = getStaticIndex(loop.getUpperBound());
  std::optional<int64_t> step = getStaticIndex(loop.getStep());
  if (!lower || !upper || !step || *step <= 0 || *upper <= *lower)
    return std::nullopt;
  int64_t span, biased, lastOffset, lastIndex;
  if (llvm::SubOverflow(*upper, *lower, span) || llvm::AddOverflow(span, *step - 1, biased))
    return std::nullopt;
  int64_t trip = biased / *step;
  if (trip > kMaxUnrolledTrip || llvm::MulOverflow(trip - 1, *step, lastOffset) ||
      llvm::AddOverflow(*lower, lastOffset, lastIndex))
    return std::nullopt;
  return trip;
}

bool isSingleLaneAccLoop(scf::ForOp loop) {
  if (loop.getNumRegionIterArgs() != 1)
    return false;
  auto accumulator = dyn_cast<AccType>(loop.getRegionIterArgs().front().getType());
  if (!accumulator || !isSingleLaneAccumulator(accumulator))
    return false;
  bool hasMac = false;
  loop.getBody()->walk([&](MacOp) { hasMac = true; });
  return hasMac;
}

void unrollAccLoop(scf::ForOp loop, int64_t lower, int64_t step, int64_t trip) {
  OpBuilder builder(loop);
  Location loc = loop.getLoc();
  Value carried = loop.getInitArgs().front();
  auto yield = cast<scf::YieldOp>(loop.getBody()->getTerminator());
  for (int64_t iteration = 0; iteration < trip; ++iteration) {
    IRMapping mapping;
    mapping.map(loop.getInductionVar(),
                builder.create<arith::ConstantIndexOp>(loc, lower + iteration * step));
    mapping.map(loop.getRegionIterArgs().front(), carried);
    for (Operation &op : loop.getBody()->without_terminator())
      builder.clone(op, mapping);
    carried = mapping.lookupOrDefault(yield.getOperand(0));
  }
  loop.getResult(0).replaceAllUsesWith(carried);
  loop.erase();
}

struct UnrollOndspFixedMacLoops final
    : ondrix::impl::UnrollOndspFixedMacLoopsBase<UnrollOndspFixedMacLoops> {
  void runOnOperation() override {
    // The default post-order walk lists inner loops first, so a candidate is
    // never erased by an enclosing candidate's rewrite before its own turn.
    SmallVector<scf::ForOp> candidates;
    getOperation()->walk([&](scf::ForOp loop) {
      if (isSingleLaneAccLoop(loop) && getUnrollableTripCount(loop))
        candidates.push_back(loop);
    });
    for (scf::ForOp loop : candidates) {
      int64_t lower = *getStaticIndex(loop.getLowerBound());
      int64_t step = *getStaticIndex(loop.getStep());
      unrollAccLoop(loop, lower, step, *getUnrollableTripCount(loop));
    }
  }
};

} // namespace

std::unique_ptr<Pass> ondrix::createUnrollOndspFixedMacLoopsPass() {
  return std::make_unique<UnrollOndspFixedMacLoops>();
}
