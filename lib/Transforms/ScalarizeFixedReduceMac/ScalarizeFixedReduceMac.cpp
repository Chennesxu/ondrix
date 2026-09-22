#include "ondrix/Transforms/Passes.h"

#include "ondrix/Analysis/FixedPointPrefixRangeAnalysis.h"
#include "ondrix/Analysis/ReductionWindowAnalysis.h"
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
#define GEN_PASS_DEF_SCALARIZEONDSPCERTIFIEDCONSTANTREDUCE
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

/// Whether the rewrite below would actually put this reduction in
/// straight-line form. The budget prescan must ask exactly this: pricing a
/// reduction it will then skip declines work that was never going to be
/// emitted, and the unroll pass downstream never sees those terms at all.
bool isStraightLineCandidate(ReduceMacOp reduce) {
  auto accumulator = llvm::dyn_cast<AccType>(reduce.getInitial().getType());
  auto numeric = llvm::dyn_cast<FixedAttr>(reduce.getNumeric());
  return accumulator && numeric && reduce.getProduct() && isSingleLaneAccumulator(accumulator) &&
         isRankOneMemRefOf(reduce.getLhs().getType(), numeric.getStorage()) &&
         isRankOneMemRefOf(reduce.getRhs().getType(), numeric.getStorage());
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

/// Functions the straight-line form is declined for; every reduction there
/// keeps the loop so one function never mixes the two shapes. Both passes
/// price the same candidates, so their verdicts agree on every function. Two
/// independent quantities decline it: the per-function TERM total, which buys
/// instruction count with instruction cache, and one reduction's CARRIED
/// WINDOW, which the term total cannot see -- a 64-term dot carries nothing
/// while a 16-tap filter carries fifteen values across its sample loop.
llvm::DenseSet<Operation *> collectOverBudgetFunctions(ModuleOp module, int64_t maxUnrolledTerms,
                                                       int64_t maxCarriedWindow) {
  llvm::DenseSet<Operation *> overBudget;
  if (maxUnrolledTerms <= 0 && maxCarriedWindow <= 0)
    return overBudget;
  llvm::DenseMap<Operation *, int64_t> totals;
  module.walk([&](ReduceMacOp op) {
    auto function = op->getParentOfType<func::FuncOp>();
    if (!function)
      return;
    if (!isStraightLineCandidate(op))
      return;
    if (maxCarriedWindow > 0) {
      // A window this analysis cannot read declines the straight-line form
      // rather than being assumed to carry nothing.
      std::optional<int64_t> carried = ondrix::analysis::getStraightLineCarriedWindow(op);
      if (!carried || *carried > maxCarriedWindow)
        overBudget.insert(function);
    }
    if (std::optional<int64_t> terms = getStraightLineTerms(op))
      totals[function] += *terms;
  });
  if (maxUnrolledTerms > 0)
    for (auto [function, total] : totals)
      if (total > maxUnrolledTerms)
        overBudget.insert(function);
  return overBudget;
}

/// Carrier of the certified chain: wide enough for any storage the match
/// admits, and a register pair on the 32-bit targets this route serves.
constexpr unsigned kCarrierWidth = 64;
/// Products one loop iteration unrolls; past it the widening add is amortized
/// and the body only grows.
constexpr int64_t kMaxLoopGroup = 16;

/// A constant-coefficient reduction the certificates admit, with the
/// coefficients in read order and the certified group width.
struct CertifiedReduction {
  ReduceMacOp reduce;
  SmallVector<llvm::APInt> coefficients;
  int64_t terms;
  int64_t group;
  unsigned termWidth;
};

std::optional<CertifiedReduction> matchCertifiedReduction(ReduceMacOp reduce, int64_t maxElements,
                                                          unsigned registerWidth) {
  auto accumulator = llvm::dyn_cast<AccType>(reduce.getInitial().getType());
  auto numeric = llvm::dyn_cast<FixedAttr>(reduce.getNumeric());
  if (!accumulator || !numeric || !reduce.getProduct() || !isSingleLaneAccumulator(accumulator) ||
      !isFullProduct(*reduce.getProduct()) || reduce.getProduct()->getShift() != 0 ||
      numeric.getSignedness() != Signedness::Signed ||
      accumulator.getSignedness() != Signedness::Signed)
    return std::nullopt;
  auto storage = llvm::dyn_cast<IntegerType>(numeric.getStorage());
  auto accumulatorStorage = llvm::dyn_cast<IntegerType>(accumulator.getStorage());
  if (!storage || !accumulatorStorage || !accumulatorStorage.isSignless() ||
      2 * storage.getWidth() > kCarrierWidth || accumulator.getFrac() != 2 * numeric.getFrac() ||
      accumulatorStorage.getWidth() > kCarrierWidth)
    return std::nullopt;
  // The group is summed as narrow as the exact product allows, but never
  // narrower than one machine register: below that width a group shares no
  // carrier-width add and the narrowing is pure cost.
  unsigned termWidth =
      std::max<unsigned>(2 * storage.getWidth(), std::min<unsigned>(kCarrierWidth, registerWidth));
  if (!reduce.getInitial().getDefiningOp<AccZeroOp>() ||
      !llvm::all_of(reduce.getResult().getUsers(),
                    [](Operation *user) { return llvm::isa<AccExportOp>(user); }))
    return std::nullopt;
  if (!isRankOneMemRefOf(reduce.getLhs().getType(), storage) ||
      !isRankOneMemRefOf(reduce.getRhs().getType(), storage))
    return std::nullopt;
  std::optional<int64_t> lhsExtent = getStaticExtent(reduce.getLhs());
  std::optional<int64_t> rhsExtent = getStaticExtent(reduce.getRhs());
  if (!lhsExtent || !rhsExtent || *lhsExtent != *rhsExtent || *lhsExtent < 2 ||
      *lhsExtent > maxElements)
    return std::nullopt;
  std::optional<SmallVector<llvm::APInt>> coefficients =
      ondrix::conversion::getConstantCoefficientsInReadOrder(reduce.getRhs(), *lhsExtent,
                                                             maxElements);
  if (!coefficients)
    return std::nullopt;
  if (accumulator.getUpdateOverflow() == OverflowMode::Saturate &&
      failed(
          ondrix::analysis::FixedPointPrefixRangePlanner::proveOrderedZeroSeededConstantReduction(
              reduce, *coefficients)))
    return std::nullopt;
  int64_t group = ondrix::analysis::FixedPointPrefixRangePlanner::largestCertifiedTermGroup(
      reduce, *coefficients, termWidth);
  return CertifiedReduction{reduce, std::move(*coefficients), *lhsExtent,
                            std::max<int64_t>(1, group), termWidth};
}

/// One certified group: `count` products from `first` (a compile-time index
/// when `base` is null, else `base + j` with the coefficient loaded), summed
/// in the term width and added to the wrapping accumulator.
Value emitCertifiedGroup(OpBuilder &builder, Location loc, const CertifiedReduction &certified,
                         Value accumulator, Value base, int64_t first, int64_t count,
                         FixedAttr termNumeric) {
  IntegerType termType = builder.getIntegerType(certified.termWidth);
  ReduceMacOp reduce = certified.reduce;
  Value sum;
  for (int64_t offset = 0; offset < count; ++offset) {
    Value index = builder.create<arith::ConstantIndexOp>(loc, base ? offset : first + offset);
    if (base)
      index = builder.create<arith::AddIOp>(loc, base, index);
    Value sample = builder.create<memref::LoadOp>(loc, reduce.getLhs(), index);
    Value wideSample = builder.create<arith::ExtSIOp>(loc, termType, sample);
    Value wideCoefficient;
    if (base) {
      Value coefficient = builder.create<memref::LoadOp>(loc, reduce.getRhs(), index);
      wideCoefficient = builder.create<arith::ExtSIOp>(loc, termType, coefficient);
    } else {
      wideCoefficient = builder.create<arith::ConstantIntOp>(
          loc, certified.coefficients[first + offset].getSExtValue(), termType);
    }
    Value product = builder.create<arith::MulIOp>(loc, wideSample, wideCoefficient);
    sum = sum ? builder.create<arith::AddIOp>(loc, sum, product).getResult() : product;
  }
  return builder.create<AccAddTermOp>(loc, accumulator.getType(), accumulator, sum, termNumeric);
}

struct ScalarizeOndspCertifiedConstantReduce final
    : ondrix::impl::ScalarizeOndspCertifiedConstantReduceBase<
          ScalarizeOndspCertifiedConstantReduce> {
  void runOnOperation() override {
    llvm::DenseSet<Operation *> overBudget =
        collectOverBudgetFunctions(getOperation(), maxUnrolledTerms, maxCarriedWindow);
    SmallVector<ReduceMacOp> candidates;
    getOperation()->walk([&](ReduceMacOp op) { candidates.push_back(op); });
    for (ReduceMacOp reduce : candidates) {
      std::optional<CertifiedReduction> certified =
          matchCertifiedReduction(reduce, maxElements, unsigned(scalarRegisterBits));
      if (!certified)
        continue;
      bool straightLine = certified->terms <= kMaxUnrolledLength &&
                          !overBudget.contains(reduce->getParentOfType<func::FuncOp>());
      // On a machine whose registers already hold the carrier the narrow term
      // buys nothing, so the only thing left to win is the per-term loop
      // overhead a block amortizes. A reduction the budget already
      // straight-lined has none, and its definitional expansion is the better
      // form; the sibling pass prices the same candidates, so the two agree on
      // which reductions those are.
      if (straightLine && scalarRegisterBits >= int64_t(kCarrierWidth))
        continue;
      auto declared = llvm::cast<AccType>(reduce.getInitial().getType());
      auto numeric = llvm::cast<FixedAttr>(reduce.getNumeric());
      OpBuilder builder(reduce);
      Location loc = reduce.getLoc();
      // The certificate bounds the exact sum within the declared storage, so
      // the 64-bit carrier holds it exactly and a 32-bit target adds it as one
      // carry pair, without the narrower width's sign fix-ups at every use.
      auto wrapping = AccType::get(declared.getContext(), builder.getIntegerType(kCarrierWidth),
                                   declared.getFrac(), declared.getSignedness(), OverflowMode::Wrap,
                                   declared.getLanes());
      auto termNumeric =
          FixedAttr::get(numeric.getContext(), Signedness::Signed,
                         builder.getIntegerType(certified->termWidth), declared.getFrac());
      Value chain = builder.create<AccZeroOp>(loc, wrapping);

      int64_t group = straightLine ? certified->group : std::min(certified->group, kMaxLoopGroup);
      int64_t fullGroups = certified->terms / group;
      int64_t covered = fullGroups * group;
      if (straightLine) {
        for (int64_t first = 0; first < covered; first += group)
          chain = emitCertifiedGroup(builder, loc, *certified, chain, Value(), first, group,
                                     termNumeric);
      } else if (fullGroups > 0) {
        Value lower = builder.create<arith::ConstantIndexOp>(loc, 0);
        Value upper = builder.create<arith::ConstantIndexOp>(loc, covered);
        Value step = builder.create<arith::ConstantIndexOp>(loc, group);
        auto loop = builder.create<scf::ForOp>(
            loc, lower, upper, step, ValueRange{chain},
            [&](OpBuilder &body, Location bodyLoc, Value base, ValueRange iterArgs) {
              Value next = emitCertifiedGroup(body, bodyLoc, *certified, iterArgs.front(), base, 0,
                                              group, termNumeric);
              body.create<scf::YieldOp>(bodyLoc, next);
            });
        chain = loop.getResult(0);
      }
      // The certificate covers the products past the last full group singly.
      for (int64_t term = covered; term < certified->terms; ++term)
        chain = emitCertifiedGroup(builder, loc, *certified, chain, Value(), term, 1, termNumeric);
      Operation *declaredSeed = reduce.getInitial().getDefiningOp();
      reduce.getResult().replaceAllUsesWith(chain);
      reduce.erase();
      if (declaredSeed->use_empty())
        declaredSeed->erase();
    }
  }
};

struct ScalarizeOndspFixedReduceMac final
    : ondrix::impl::ScalarizeOndspFixedReduceMacBase<ScalarizeOndspFixedReduceMac> {
  llvm::DenseSet<Operation *> overBudget;

  void runOnOperation() override {
    overBudget = collectOverBudgetFunctions(getOperation(), maxUnrolledTerms, maxCarriedWindow);
    SmallVector<ReduceMacOp> candidates;
    getOperation()->walk([&](ReduceMacOp op) { candidates.push_back(op); });
    for (ReduceMacOp reduce : candidates) {
      if (!isStraightLineCandidate(reduce))
        continue;
      auto numeric = llvm::cast<FixedAttr>(reduce.getNumeric());

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

std::unique_ptr<Pass> ondrix::createScalarizeOndspCertifiedConstantReducePass() {
  return std::make_unique<ScalarizeOndspCertifiedConstantReduce>();
}
