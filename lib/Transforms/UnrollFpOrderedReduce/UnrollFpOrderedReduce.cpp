#include "ondrix/Transforms/Passes.h"

#include "ondrix/Dialect/ondsp/IR/OndspDialect.h"
#include "ondrix/Dialect/ondsp/IR/OndspOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/MathExtras.h"

namespace ondrix {
#define GEN_PASS_DEF_UNROLLONDSPFPORDEREDREDUCE
#include "ondrix/Transforms/Passes.h.inc"
} // namespace ondrix

using namespace mlir;
using namespace ondrix::ondsp;

namespace {

/// A contract that fixes the fold order, so replaying the body at constant
/// induction values reproduces the declared sequence exactly. `fast` is
/// excluded: its sites carry spent-permission records this pass must not
/// replicate, and the fast reduction pass owns their shape.
bool isOrderDeclared(FpContractMode contract) {
  return contract == FpContractMode::Off || contract == FpContractMode::Fma;
}

/// A compile-time index, folded through the index arithmetic a producing pass
/// leaves behind: the fast reduction's group bounds are `upper - upper % C`
/// built from constants, and nothing canonicalizes between that pass and this
/// one. Only all-constant operands fold, so the answer is exact where it is
/// given at all; a division by zero or an overflow is refused.
std::optional<int64_t> getStaticIndex(Value value, int depth = 0) {
  if (std::optional<int64_t> constant = getConstantIntValue(value))
    return constant;
  Operation *op = value.getDefiningOp();
  if (depth >= 8 || !op || op->getNumOperands() != 2)
    return std::nullopt;
  std::optional<int64_t> lhs = getStaticIndex(op->getOperand(0), depth + 1);
  std::optional<int64_t> rhs = getStaticIndex(op->getOperand(1), depth + 1);
  if (!lhs || !rhs)
    return std::nullopt;
  int64_t result;
  if (isa<arith::AddIOp>(op))
    return llvm::AddOverflow(*lhs, *rhs, result) ? std::nullopt : std::optional(result);
  if (isa<arith::SubIOp>(op))
    return llvm::SubOverflow(*lhs, *rhs, result) ? std::nullopt : std::optional(result);
  if (isa<arith::MulIOp>(op))
    return llvm::MulOverflow(*lhs, *rhs, result) ? std::nullopt : std::optional(result);
  if (isa<arith::RemUIOp>(op) && *lhs >= 0 && *rhs > 0)
    return *lhs % *rhs;
  if (isa<arith::DivUIOp>(op) && *lhs >= 0 && *rhs > 0)
    return *lhs / *rhs;
  return std::nullopt;
}

/// Static extent of a rank-1 f32 memref operand, resolved through the casts
/// bufferization inserts to state the runtime equal-length contract.
std::optional<int64_t> getStaticExtent(Value operand) {
  while (true) {
    auto type = llvm::dyn_cast<MemRefType>(operand.getType());
    if (type && type.getRank() == 1 && !type.isDynamicDim(0))
      return type.getDimSize(0);
    auto castOp = operand.getDefiningOp<memref::CastOp>();
    if (!castOp)
      return std::nullopt;
    operand = castOp.getSource();
  }
}

bool isRankOneF32MemRef(Type type) {
  auto memref = llvm::dyn_cast<MemRefType>(type);
  return memref && memref.getRank() == 1 && memref.getElementType().isF32();
}

bool isStraightLineCandidate(ReduceMacOp reduce) {
  auto numeric = llvm::dyn_cast<FpAttr>(reduce.getNumeric());
  return numeric && numeric.getFormat().isF32() && isOrderDeclared(numeric.getContract()) &&
         !reduce.getProduct() && reduce.getInitial().getType().isF32() &&
         reduce.getResult().getType().isF32() && isRankOneF32MemRef(reduce.getLhs().getType()) &&
         isRankOneF32MemRef(reduce.getRhs().getType());
}

/// Terms the straight-line form would emit, or nothing when the reduction
/// does not qualify for that form at all.
std::optional<int64_t> getStraightLineTerms(ReduceMacOp reduce, int64_t maxTerms) {
  std::optional<int64_t> lhs = getStaticExtent(reduce.getLhs());
  std::optional<int64_t> rhs = getStaticExtent(reduce.getRhs());
  if (!lhs || !rhs || *lhs != *rhs || *lhs > maxTerms)
    return std::nullopt;
  return *lhs;
}

/// An accumulator loop a lowering left behind: every iteration argument an f32
/// accumulator, no nested loop, and a body that advances them. The FIR family's
/// tap loop carries one; the fast reduction's chain loop carries its interleave
/// count, and unrolling it is the same identity -- the permission its final
/// fold records is already spent, and the record is a set, so replaying the
/// body neither spends nor double-counts one.
bool isF32AccLoop(scf::ForOp loop) {
  if (loop.getNumRegionIterArgs() < 1)
    return false;
  for (Value argument : loop.getRegionIterArgs())
    if (!argument.getType().isF32())
      return false;
  if (loop.getBody()->walk([](scf::ForOp) { return WalkResult::interrupt(); }).wasInterrupted())
    return false;
  return loop.getBody()
      ->walk([](Operation *op) {
        return isa<arith::AddFOp, math::FmaOp>(op) ? WalkResult::interrupt()
                                                   : WalkResult::advance();
      })
      .wasInterrupted();
}

// Every intermediate here can overflow i64 for legal extreme bounds; anything
// that overflows is refused instead of wrapping into a wrong trip count.
std::optional<int64_t> getUnrollableTripCount(scf::ForOp loop, int64_t maxTerms) {
  std::optional<int64_t> lower = getStaticIndex(loop.getLowerBound());
  std::optional<int64_t> upper = getStaticIndex(loop.getUpperBound());
  std::optional<int64_t> step = getStaticIndex(loop.getStep());
  if (!lower || !upper || !step || *step <= 0 || *upper <= *lower)
    return std::nullopt;
  int64_t span, biased, lastOffset, lastIndex;
  if (llvm::SubOverflow(*upper, *lower, span) || llvm::AddOverflow(span, *step - 1, biased))
    return std::nullopt;
  int64_t trip = biased / *step;
  if (trip > maxTerms || llvm::MulOverflow(trip - 1, *step, lastOffset) ||
      llvm::AddOverflow(*lower, lastOffset, lastIndex))
    return std::nullopt;
  return trip;
}

void unrollAccLoop(scf::ForOp loop, int64_t lower, int64_t step, int64_t trip) {
  OpBuilder builder(loop);
  Location loc = loop.getLoc();
  SmallVector<Value> carried(loop.getInitArgs().begin(), loop.getInitArgs().end());
  auto yield = cast<scf::YieldOp>(loop.getBody()->getTerminator());
  for (int64_t iteration = 0; iteration < trip; ++iteration) {
    IRMapping mapping;
    mapping.map(loop.getInductionVar(),
                builder.create<arith::ConstantIndexOp>(loc, lower + iteration * step));
    for (auto [argument, value] : llvm::zip(loop.getRegionIterArgs(), carried))
      mapping.map(argument, value);
    for (Operation &op : loop.getBody()->without_terminator())
      builder.clone(op, mapping);
    for (auto [slot, operand] : llvm::zip(carried, yield.getOperands()))
      slot = mapping.lookupOrDefault(operand);
  }
  for (auto [result, value] : llvm::zip(loop.getResults(), carried))
    result.replaceAllUsesWith(value);
  loop.erase();
}

struct UnrollOndspFpOrderedReduce final
    : ondrix::impl::UnrollOndspFpOrderedReduceBase<UnrollOndspFpOrderedReduce> {
  /// Functions whose straight-line total exceeds the budget; everything there
  /// keeps its loop so one function never mixes the two shapes.
  llvm::DenseSet<Operation *> overBudget;

  void collectOverBudgetFunctions() {
    if (maxUnrolledTerms <= 0)
      return;
    llvm::DenseMap<Operation *, int64_t> totals;
    getOperation()->walk([&](Operation *op) {
      auto function = op->getParentOfType<func::FuncOp>();
      if (!function)
        return;
      if (auto reduce = dyn_cast<ReduceMacOp>(op)) {
        if (vectorWidth <= 1 && isStraightLineCandidate(reduce))
          if (std::optional<int64_t> terms = getStraightLineTerms(reduce, maxStraightLineTerms))
            totals[function] += *terms;
        return;
      }
      if (auto loop = dyn_cast<scf::ForOp>(op))
        if (isF32AccLoop(loop))
          if (std::optional<int64_t> trip = getUnrollableTripCount(loop, maxStraightLineTerms))
            totals[function] += *trip * loop.getNumRegionIterArgs();
    });
    for (auto [function, total] : totals)
      if (total > maxUnrolledTerms)
        overBudget.insert(function);
  }

  /// Replace one ordered reduction by the straight-line chain its contract
  /// denotes: product then fold, in increasing index order, onto the same
  /// initial accumulator. This is the scalar lowering's own body replayed at
  /// constant indices, so every rounding stays where it was.
  void straightenReduction(ReduceMacOp reduce, int64_t terms) {
    OpBuilder builder(reduce);
    Location loc = reduce.getLoc();
    auto numeric = llvm::cast<FpAttr>(reduce.getNumeric());
    Value chain = reduce.getInitial();
    for (int64_t term = 0; term < terms; ++term) {
      Value index = builder.create<arith::ConstantIndexOp>(loc, term);
      Value sample = builder.create<memref::LoadOp>(loc, reduce.getLhs(), index);
      Value coefficient = builder.create<memref::LoadOp>(loc, reduce.getRhs(), index);
      if (numeric.getContract() == FpContractMode::Fma) {
        chain = builder.create<math::FmaOp>(loc, sample, coefficient, chain);
        continue;
      }
      Value product = builder.create<arith::MulFOp>(loc, sample, coefficient);
      chain = builder.create<arith::AddFOp>(loc, chain, product);
    }
    reduce.getResult().replaceAllUsesWith(chain);
    reduce.erase();
  }

  void runOnOperation() override {
    collectOverBudgetFunctions();

    // Above one lane the lane-blocked scalar lowering owns the ordered
    // reductions; only the accumulator loops, which no lane stage claims,
    // are taken here.
    SmallVector<ReduceMacOp> reductions;
    if (vectorWidth <= 1)
      getOperation()->walk([&](ReduceMacOp op) { reductions.push_back(op); });
    for (ReduceMacOp reduce : reductions) {
      if (!isStraightLineCandidate(reduce) ||
          overBudget.contains(reduce->getParentOfType<func::FuncOp>()))
        continue;
      if (std::optional<int64_t> terms = getStraightLineTerms(reduce, maxStraightLineTerms))
        straightenReduction(reduce, *terms);
    }

    // The default post-order walk lists inner loops first, so a candidate is
    // never erased by an enclosing candidate's rewrite before its own turn.
    SmallVector<scf::ForOp> loops;
    getOperation()->walk([&](scf::ForOp loop) {
      if (isF32AccLoop(loop) && getUnrollableTripCount(loop, maxStraightLineTerms))
        loops.push_back(loop);
    });
    for (scf::ForOp loop : loops) {
      if (overBudget.contains(loop->getParentOfType<func::FuncOp>()))
        continue;
      unrollAccLoop(loop, *getStaticIndex(loop.getLowerBound()), *getStaticIndex(loop.getStep()),
                    *getUnrollableTripCount(loop, maxStraightLineTerms));
    }
  }
};

} // namespace

std::unique_ptr<Pass> ondrix::createUnrollOndspFpOrderedReducePass() {
  return std::make_unique<UnrollOndspFpOrderedReduce>();
}
