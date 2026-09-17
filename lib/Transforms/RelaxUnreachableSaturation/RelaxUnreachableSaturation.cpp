#include "ondrix/Transforms/Passes.h"

#include "ondrix/Dialect/ondsp/IR/OndspDialect.h"
#include "ondrix/Dialect/ondsp/IR/OndspOps.h"
#include "ondrix/Dialect/ondsp/IR/OndspTypes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/DenseMap.h"

namespace ondrix {
#define GEN_PASS_DEF_RELAXONDSPUNREACHABLESATURATION
#include "ondrix/Transforms/Passes.h.inc"
} // namespace ondrix

using namespace mlir;

namespace {

// Wide enough that a chain of i64 sums of i16 products never approaches it;
// every step still checks, so the width is a budget and not an assumption.
constexpr unsigned kAnalysisWidth = 160;
constexpr unsigned kMaxDepth = 256;

struct Bounds {
  llvm::APInt lower;
  llvm::APInt upper;
};

static Bounds signedRange(unsigned width) {
  return {llvm::APInt::getSignedMinValue(width).sext(kAnalysisWidth),
          llvm::APInt::getSignedMaxValue(width).sext(kAnalysisWidth)};
}

static bool fitsSigned(const Bounds &bounds, unsigned width) {
  Bounds range = signedRange(width);
  return bounds.lower.sge(range.lower) && bounds.upper.sle(range.upper);
}

static Bounds clampSigned(const Bounds &bounds, unsigned width) {
  Bounds range = signedRange(width);
  return {bounds.lower.slt(range.lower) ? range.lower : bounds.lower,
          bounds.upper.sgt(range.upper) ? range.upper : bounds.upper};
}

static unsigned integerWidthOrZero(Type type) {
  auto integer = dyn_cast<IntegerType>(type);
  return integer && integer.isSignless() ? integer.getWidth() : 0;
}

/// A conservative signed interval of `value`. Leaves and unrecognized
/// producers fall back to their type's full signed range, so the walk is
/// total; every arithmetic step that could leave its own result type falls
/// back the same way, which is what makes a wrapping operation safe to cross.
class BoundsAnalysis {
public:
  std::optional<Bounds> compute(Value value) { return visit(value, 0); }

private:
  std::optional<Bounds> visit(Value value, unsigned depth);

  static Bounds add(const Bounds &lhs, const Bounds &rhs) {
    return {lhs.lower + rhs.lower, lhs.upper + rhs.upper};
  }

  // The shifted reading is bounded below by the floor of the lower end; any
  // declared rounding moves a result up by at most one.
  static Bounds shiftRight(const Bounds &bounds, unsigned shift) {
    return {bounds.lower.ashr(shift), bounds.upper.ashr(shift) + 1};
  }

  std::optional<Bounds> visitLoopResult(scf::ForOp loop, unsigned index, unsigned depth);

  std::optional<Bounds> fallback(Value value) {
    unsigned width = integerWidthOrZero(value.getType());
    if (width == 0 || width > kAnalysisWidth)
      return std::nullopt;
    return signedRange(width);
  }

  llvm::DenseMap<Value, Bounds> cache;
};

// A counted accumulator loop: every iteration adds one term that does not
// read the accumulator, so the whole loop is the initial value plus the trip
// count times one term's interval. That interval must straddle zero, which is
// what makes the N-step bound cover every prefix and not only the last one.
std::optional<Bounds> BoundsAnalysis::visitLoopResult(scf::ForOp loop, unsigned index,
                                                      unsigned depth) {
  std::optional<int64_t> first = getConstantIntValue(loop.getLowerBound());
  std::optional<int64_t> last = getConstantIntValue(loop.getUpperBound());
  std::optional<int64_t> stride = getConstantIntValue(loop.getStep());
  if (!first || !last || !stride || *stride <= 0 || *last < *first)
    return std::nullopt;
  int64_t trips = (*last - *first + *stride - 1) / *stride;

  auto yielded = loop.getBody()->getTerminator()->getOperand(index);
  auto term = yielded.getDefiningOp<ondrix::ondsp::AccAddTermOp>();
  if (!term || term.getAcc() != loop.getRegionIterArgs()[index])
    return std::nullopt;
  auto type = cast<ondrix::ondsp::AccType>(term.getType());
  unsigned width = integerWidthOrZero(type.getStorage());
  std::optional<Bounds> initial = visit(loop.getInitArgs()[index], depth + 1);
  std::optional<Bounds> step = visit(term.getTerm(), depth + 1);
  if (!initial || !step || width == 0)
    return std::nullopt;
  llvm::APInt zero(kAnalysisWidth, 0);
  if (step->lower.sgt(zero) || step->upper.slt(zero))
    return std::nullopt;

  llvm::APInt count(kAnalysisWidth, trips, /*isSigned=*/true);
  Bounds total{initial->lower + step->lower * count, initial->upper + step->upper * count};
  if (fitsSigned(total, width))
    return total;
  if (type.getUpdateOverflow() == ondrix::ondsp::OverflowMode::Saturate)
    return clampSigned(total, width);
  return signedRange(width);
}

std::optional<Bounds> BoundsAnalysis::visit(Value value, unsigned depth) {
  auto cached = cache.find(value);
  if (cached != cache.end())
    return cached->second;
  if (depth > kMaxDepth)
    return fallback(value);

  Operation *producer = value.getDefiningOp();
  std::optional<Bounds> result;
  if (!producer) {
    result = fallback(value);
  } else if (auto constant = dyn_cast<arith::ConstantOp>(producer)) {
    if (auto attr = dyn_cast<IntegerAttr>(constant.getValue())) {
      llvm::APInt exact = attr.getValue().sext(kAnalysisWidth);
      result = Bounds{exact, exact};
    }
  } else if (auto extend = dyn_cast<arith::ExtSIOp>(producer)) {
    result = visit(extend.getIn(), depth + 1);
  } else if (auto truncate = dyn_cast<arith::TruncIOp>(producer)) {
    std::optional<Bounds> input = visit(truncate.getIn(), depth + 1);
    unsigned width = integerWidthOrZero(truncate.getType());
    if (input && width != 0 && fitsSigned(*input, width))
      result = *input;
  } else if (auto sum = dyn_cast<arith::AddIOp>(producer)) {
    std::optional<Bounds> lhs = visit(sum.getLhs(), depth + 1);
    std::optional<Bounds> rhs = visit(sum.getRhs(), depth + 1);
    unsigned width = integerWidthOrZero(sum.getType());
    if (lhs && rhs && width != 0) {
      Bounds exact = add(*lhs, *rhs);
      if (fitsSigned(exact, width))
        result = exact;
    }
  } else if (auto difference = dyn_cast<arith::SubIOp>(producer)) {
    std::optional<Bounds> lhs = visit(difference.getLhs(), depth + 1);
    std::optional<Bounds> rhs = visit(difference.getRhs(), depth + 1);
    unsigned width = integerWidthOrZero(difference.getType());
    if (lhs && rhs && width != 0) {
      Bounds exact{lhs->lower - rhs->upper, lhs->upper - rhs->lower};
      if (fitsSigned(exact, width))
        result = exact;
    }
  } else if (auto product = dyn_cast<arith::MulIOp>(producer)) {
    std::optional<Bounds> lhs = visit(product.getLhs(), depth + 1);
    std::optional<Bounds> rhs = visit(product.getRhs(), depth + 1);
    unsigned width = integerWidthOrZero(product.getType());
    if (lhs && rhs && width != 0) {
      llvm::APInt corners[4] = {lhs->lower * rhs->lower, lhs->lower * rhs->upper,
                                lhs->upper * rhs->lower, lhs->upper * rhs->upper};
      Bounds exact{corners[0], corners[0]};
      for (const llvm::APInt &corner : corners) {
        if (corner.slt(exact.lower))
          exact.lower = corner;
        if (corner.sgt(exact.upper))
          exact.upper = corner;
      }
      if (fitsSigned(exact, width))
        result = exact;
    }
  } else if (auto zero = dyn_cast<ondrix::ondsp::AccZeroOp>(producer)) {
    llvm::APInt none(kAnalysisWidth, 0);
    result = Bounds{none, none};
  } else if (auto term = dyn_cast<ondrix::ondsp::AccAddTermOp>(producer)) {
    std::optional<Bounds> accumulator = visit(term.getAcc(), depth + 1);
    std::optional<Bounds> addend = visit(term.getTerm(), depth + 1);
    auto type = cast<ondrix::ondsp::AccType>(term.getType());
    unsigned width = integerWidthOrZero(type.getStorage());
    if (accumulator && addend && width != 0) {
      Bounds exact = add(*accumulator, *addend);
      if (fitsSigned(exact, width))
        result = exact;
      else if (type.getUpdateOverflow() == ondrix::ondsp::OverflowMode::Saturate)
        result = clampSigned(exact, width);
      else
        result = signedRange(width);
    }
  } else if (auto loop = dyn_cast<scf::ForOp>(producer)) {
    result = visitLoopResult(loop, cast<OpResult>(value).getResultNumber(), depth);
  } else if (auto exported = dyn_cast<ondrix::ondsp::AccExportOp>(producer)) {
    auto type = cast<ondrix::ondsp::AccType>(exported.getAcc().getType());
    unsigned destination = integerWidthOrZero(exported.getDst().getStorage());
    std::optional<Bounds> accumulator = visit(exported.getAcc(), depth + 1);
    if (accumulator && destination != 0 && type.getFrac() >= exported.getDst().getFrac()) {
      Bounds shifted = shiftRight(*accumulator, type.getFrac() - exported.getDst().getFrac());
      if (fitsSigned(shifted, destination))
        result = shifted;
      else if (exported.getOverflow() == ondrix::ondsp::OverflowMode::Saturate)
        result = clampSigned(shifted, destination);
      else
        result = signedRange(destination);
    }
  } else if (auto scaled = dyn_cast<ondrix::ondsp::RoundShiftOp>(producer)) {
    ondrix::ondsp::ScaleAttr scale = scaled.getScale();
    unsigned destination = integerWidthOrZero(scale.getSaturateTo());
    std::optional<Bounds> input = visit(scaled.getInput(), depth + 1);
    if (input && destination != 0 && scale.getPreShiftLeft() == 0) {
      Bounds shifted = shiftRight(*input, scale.getPostShiftRight());
      if (fitsSigned(shifted, destination))
        result = shifted;
      else if (scale.getOverflow() == ondrix::ondsp::OverflowMode::Saturate)
        result = clampSigned(shifted, destination);
      else
        result = signedRange(destination);
    }
  }

  if (!result)
    result = fallback(value);
  if (result)
    cache.try_emplace(value, *result);
  return result;
}

class RelaxOndspUnreachableSaturationPass final
    : public ondrix::impl::RelaxOndspUnreachableSaturationBase<
          RelaxOndspUnreachableSaturationPass> {
public:
  using ondrix::impl::RelaxOndspUnreachableSaturationBase<
      RelaxOndspUnreachableSaturationPass>::RelaxOndspUnreachableSaturationBase;

  void runOnOperation() override {
    BoundsAnalysis analysis;
    getOperation().walk([&](ondrix::ondsp::RoundShiftOp op) {
      ondrix::ondsp::ScaleAttr scale = op.getScale();
      if (scale.getOverflow() != ondrix::ondsp::OverflowMode::Saturate ||
          scale.getPreShiftLeft() != 0)
        return;
      unsigned destination = integerWidthOrZero(scale.getSaturateTo());
      if (destination == 0)
        return;
      std::optional<Bounds> input = analysis.compute(op.getInput());
      if (!input || !fitsSigned(shiftedReading(*input, scale.getPostShiftRight()), destination))
        return;
      op.setScaleAttr(ondrix::ondsp::ScaleAttr::get(
          op.getContext(), scale.getPreShiftLeft(), scale.getPostShiftRight(), scale.getRounding(),
          ondrix::ondsp::OverflowMode::Wrap, scale.getSaturateTo()));
    });
  }

private:
  static Bounds shiftedReading(const Bounds &bounds, unsigned shift) {
    return {bounds.lower.ashr(shift), bounds.upper.ashr(shift) + 1};
  }
};

} // namespace

std::unique_ptr<Pass> ondrix::createRelaxOndspUnreachableSaturationPass() {
  return std::make_unique<RelaxOndspUnreachableSaturationPass>();
}
