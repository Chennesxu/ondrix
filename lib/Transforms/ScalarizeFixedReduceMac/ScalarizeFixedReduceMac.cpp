#include "ondrix/Transforms/Passes.h"

#include "ondrix/Conversion/Utils/ReductionUtils.h"
#include "ondrix/Dialect/ondsp/IR/OndspDialect.h"
#include "ondrix/Dialect/ondsp/IR/OndspOps.h"
#include "ondrix/Dialect/ondsp/IR/OndspSemantics.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"

namespace ondrix {
#define GEN_PASS_DEF_SCALARIZEONDSPFIXEDREDUCEMAC
#include "ondrix/Transforms/Passes.h.inc"
} // namespace ondrix

using namespace mlir;
using namespace ondrix::ondsp;

namespace {

/// Longest straight-line chain one reduction may become; past it the loop
/// form is cheaper even before the per-function budget applies.
constexpr int64_t kMaxUnrolledLength = 64;

bool isRankOneMemRefOf(Type type, Type element) {
  auto memref = llvm::dyn_cast<MemRefType>(type);
  return memref && memref.getRank() == 1 && memref.getElementType() == element;
}

/// Static extent of a rank-1 memref operand, resolved through the casts
/// bufferization inserts to state the runtime equal-length contract: the
/// erased size is still the one the allocation was built with.
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

/// Terms the straight-line form would emit for this reduction, or nothing
/// when it does not qualify for that form at all.
std::optional<int64_t> getStraightLineTerms(ReduceMacOp reduce) {
  std::optional<int64_t> lhsExtent = getStaticExtent(reduce.getLhs());
  std::optional<int64_t> rhsExtent = getStaticExtent(reduce.getRhs());
  if (!lhsExtent || !rhsExtent || *lhsExtent != *rhsExtent || *lhsExtent > kMaxUnrolledLength)
    return std::nullopt;
  return *lhsExtent;
}

struct ScalarizeOndspFixedReduceMac final
    : ondrix::impl::ScalarizeOndspFixedReduceMacBase<ScalarizeOndspFixedReduceMac> {
  /// Functions whose straight-line total exceeds the budget; every reduction
  /// there keeps the loop form so one function never mixes the two shapes.
  llvm::DenseSet<Operation *> overBudget;

  void collectOverBudgetFunctions() {
    if (maxUnrolledTerms <= 0)
      return;
    llvm::DenseMap<Operation *, int64_t> totals;
    getOperation()->walk([&](ReduceMacOp op) {
      auto function = op->getParentOfType<func::FuncOp>();
      if (!function)
        return;
      if (std::optional<int64_t> terms = getStraightLineTerms(op))
        totals[function] += *terms;
    });
    for (auto [function, total] : totals)
      if (total > maxUnrolledTerms)
        overBudget.insert(function);
  }

  void runOnOperation() override {
    collectOverBudgetFunctions();
    SmallVector<ReduceMacOp> candidates;
    getOperation()->walk([&](ReduceMacOp op) { candidates.push_back(op); });
    for (ReduceMacOp reduce : candidates) {
      auto accumulator = llvm::dyn_cast<AccType>(reduce.getInitial().getType());
      auto numeric = llvm::dyn_cast<FixedAttr>(reduce.getNumeric());
      if (!accumulator || !numeric || !reduce.getProduct() ||
          !isSingleLaneAccumulator(accumulator) ||
          !isRankOneMemRefOf(reduce.getLhs().getType(), numeric.getStorage()) ||
          !isRankOneMemRefOf(reduce.getRhs().getType(), numeric.getStorage()))
        continue;

      OpBuilder builder(reduce);
      Location loc = reduce.getLoc();

      // A short static reduction unrolls to a straight-line chain: the loop
      // form pays an index update and a branch per term, which outweighs the
      // term itself (measured; the budget above is what keeps a replicated
      // lowering from spending the whole instruction cache on it).
      std::optional<int64_t> terms = getStraightLineTerms(reduce);
      if (terms && !overBudget.contains(reduce->getParentOfType<func::FuncOp>())) {
        Value chain = reduce.getInitial();
        for (int64_t term = 0; term < *terms; ++term) {
          Value index = builder.create<arith::ConstantIndexOp>(loc, term);
          Value sample = builder.create<memref::LoadOp>(loc, reduce.getLhs(), index);
          Value coefficient = builder.create<memref::LoadOp>(loc, reduce.getRhs(), index);
          chain = builder.create<MacOp>(loc, chain.getType(), chain, sample, coefficient, numeric,
                                        *reduce.getProduct());
        }
        reduce.getResult().replaceAllUsesWith(chain);
        reduce.erase();
        continue;
      }

      FailureOr<ondrix::conversion::RankOneReductionBounds> bounds =
          ondrix::conversion::createRankOneMemRefReductionBounds(
              reduce, reduce.getLhs(), reduce.getRhs(), numeric.getStorage(),
              "reduce_mac scalarization", builder);
      if (failed(bounds))
        return signalPassFailure();

      Value step = builder.create<arith::ConstantIndexOp>(loc, 1);
      auto loop = builder.create<scf::ForOp>(
          loc, bounds->lowerBound, bounds->upperBound, step, ValueRange{reduce.getInitial()},
          [&](OpBuilder &body, Location bodyLoc, Value index, ValueRange iterArgs) {
            Value sample = body.create<memref::LoadOp>(bodyLoc, reduce.getLhs(), index);
            Value coefficient = body.create<memref::LoadOp>(bodyLoc, reduce.getRhs(), index);
            Value next = body.create<MacOp>(bodyLoc, iterArgs.front().getType(), iterArgs.front(),
                                            sample, coefficient, numeric, *reduce.getProduct());
            body.create<scf::YieldOp>(bodyLoc, next);
          });
      reduce.getResult().replaceAllUsesWith(loop.getResult(0));
      reduce.erase();
    }
  }
};

} // namespace

std::unique_ptr<Pass> ondrix::createScalarizeOndspFixedReduceMacPass() {
  return std::make_unique<ScalarizeOndspFixedReduceMac>();
}
