#include "ondrix/Transforms/Passes.h"

#include "ondrix/Dialect/ondsp/IR/OndspDialect.h"
#include "ondrix/Dialect/ondsp/IR/OndspOps.h"
#include "ondrix/Dialect/ondsp/IR/OndspSemantics.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
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

/// A compile-time index value, seen directly, through `tensor.dim` of a
/// statically shaped tensor, or through `memref.dim` of one whose extent the
/// casts bufferization inserts have erased. The reduction scalarizer prices
/// the same site by the same recovery, and the two must agree or one pass
/// re-expands what the other declined for the same budget.
std::optional<int64_t> getStaticIndex(Value value) {
  if (std::optional<int64_t> constant = getConstantIntValue(value))
    return constant;
  if (auto dim = value.getDefiningOp<tensor::DimOp>()) {
    auto type = dyn_cast<RankedTensorType>(dim.getSource().getType());
    std::optional<int64_t> index = dim.getConstantIndex();
    if (!type || !index || type.isDynamicDim(*index))
      return std::nullopt;
    return type.getDimSize(*index);
  }
  auto dim = value.getDefiningOp<memref::DimOp>();
  if (!dim)
    return std::nullopt;
  std::optional<int64_t> index = dim.getConstantIndex();
  if (!index || *index < 0)
    return std::nullopt;
  Value source = dim.getSource();
  while (true) {
    auto type = dyn_cast<MemRefType>(source.getType());
    if (type && *index < type.getRank() && !type.isDynamicDim(*index))
      return type.getDimSize(*index);
    auto cast = source.getDefiningOp<memref::CastOp>();
    if (!cast)
      return std::nullopt;
    source = cast.getSource();
  }
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
  return loop.getBody()->walk([](MacOp) { return WalkResult::interrupt(); }).wasInterrupted();
}

bool isUnrollCandidate(scf::ForOp loop) {
  return isSingleLaneAccLoop(loop) && getUnrollableTripCount(loop).has_value();
}

/// Static macs an operation leaves behind once unrolling has run over
/// everything inside it. Cost is what the transform PRODUCES: a trip count
/// alone undercounts every nested candidate, which is how a 128-term budget
/// once admitted 4096. Every region is entered, because a candidate reached
/// through an `scf.if` or any other region op still multiplies.
///
/// Each operation is classified and costed once; the classification walks a
/// body, so caching it is what keeps a deep nest from being re-walked per
/// enclosing level.
struct MacCostCache {
  llvm::DenseMap<Operation *, std::optional<int64_t>> costs;
  llvm::DenseMap<Operation *, bool> candidates;

  bool isCandidate(scf::ForOp loop) {
    auto cached = candidates.find(loop);
    if (cached != candidates.end())
      return cached->second;
    bool result = isUnrollCandidate(loop);
    candidates[loop] = result;
    return result;
  }

  std::optional<int64_t> regionCost(Region &region) {
    int64_t total = 0;
    for (Block &block : region)
      for (Operation &op : block) {
        std::optional<int64_t> cost = operationCost(&op);
        if (!cost || llvm::AddOverflow(total, *cost, total))
          return std::nullopt;
      }
    return total;
  }

  std::optional<int64_t> operationCost(Operation *op) {
    auto cached = costs.find(op);
    if (cached != costs.end())
      return cached->second;
    std::optional<int64_t> result = 1;
    if (!isa<MacOp>(op)) {
      int64_t body = 0;
      for (Region &region : op->getRegions()) {
        std::optional<int64_t> cost = regionCost(region);
        if (!cost || llvm::AddOverflow(body, *cost, body)) {
          body = 0;
          result = std::nullopt;
          break;
        }
      }
      if (result) {
        result = body;
        // Only a candidate loop multiplies: a loop this pass will not unroll
        // keeps its body once, but whatever unrolls INSIDE it still expands.
        auto loop = dyn_cast<scf::ForOp>(op);
        if (loop && isCandidate(loop)) {
          int64_t total;
          if (llvm::MulOverflow(*getUnrollableTripCount(loop), body, total))
            result = std::nullopt;
          else
            result = total;
        }
      }
    }
    costs[op] = result;
    return result;
  }
};

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

    // Over-budget functions keep every loop, so one function never mixes the
    // two shapes and this pass never re-expands what the reduction scalarizer
    // declined for the same budget.
    llvm::DenseSet<Operation *> overBudget;
    if (maxUnrolledTerms > 0) {
      MacCostCache cache;
      llvm::DenseMap<Operation *, int64_t> totals;
      for (scf::ForOp loop : candidates) {
        auto function = loop->getParentOfType<func::FuncOp>();
        if (!function)
          continue;
        // Only outermost candidates are charged: a nested one is already a
        // factor of its parent's product, and charging both double counts.
        bool nested = false;
        for (Operation *parent = loop->getParentOp(); parent && !nested;
             parent = parent->getParentOp())
          if (auto enclosing = dyn_cast<scf::ForOp>(parent))
            nested = cache.isCandidate(enclosing);
        if (nested)
          continue;
        std::optional<int64_t> cost = cache.operationCost(loop);
        if (!cost || llvm::AddOverflow(totals[function], *cost, totals[function]))
          overBudget.insert(function);
      }
      for (auto [function, total] : totals)
        if (total > maxUnrolledTerms)
          overBudget.insert(function);
    }

    for (scf::ForOp loop : candidates) {
      if (overBudget.contains(loop->getParentOfType<func::FuncOp>()))
        continue;
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
