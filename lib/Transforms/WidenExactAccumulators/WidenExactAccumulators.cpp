#include "ondrix/Transforms/Passes.h"

#include "llvm/ADT/SmallVector.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

#include "ondrix/Dialect/ondsp/IR/OndspDialect.h"
#include "ondrix/Dialect/ondsp/IR/OndspOps.h"

#include <cstdint>
#include <optional>

namespace ondrix {
#define GEN_PASS_DEF_WIDENONDSPEXACTACCUMULATORS
#include "ondrix/Transforms/Passes.h.inc"
} // namespace ondrix

using namespace mlir;
using namespace ondrix::ondsp;

namespace {

// A widenable web: every accumulator-typed value from one acc_zero to one
// acc_export, plus the largest number of updates any execution can apply.
struct AccumulatorWeb {
  llvm::SmallVector<Value> values;
  int64_t maxUpdates = 0;
};

bool isFullQ15Mac(Operation *op) {
  auto numeric = [](FixedAttr attr) {
    auto storage = llvm::dyn_cast<IntegerType>(attr.getStorage());
    return storage && storage.getWidth() == 16 && attr.getFrac() == 15 &&
           attr.getSignedness() == Signedness::Signed;
  };
  if (auto mac = llvm::dyn_cast<MacOp>(op))
    return numeric(mac.getNumeric()) && mac.getProduct().getSelection() == ProductSelection::Full;
  if (auto mac = llvm::dyn_cast<MacSubOp>(op))
    return numeric(mac.getNumeric()) && mac.getProduct().getSelection() == ProductSelection::Full;
  return false;
}

Operation *soleUser(Value value) { return value.hasOneUse() ? *value.getUsers().begin() : nullptr; }

// A compile-time loop bound: a constant, or the queried extent of a
// statically shaped tensor before canonicalization folds it.
std::optional<int64_t> staticBound(Value value) {
  if (std::optional<int64_t> constant = getConstantIntValue(value))
    return constant;
  if (auto dim = value.getDefiningOp<tensor::DimOp>()) {
    std::optional<int64_t> index = getConstantIntValue(dim.getIndex());
    auto type = llvm::dyn_cast<RankedTensorType>(dim.getSource().getType());
    if (index && type && *index >= 0 && *index < type.getRank() && !type.isDynamicDim(*index))
      return type.getDimSize(*index);
  }
  return std::nullopt;
}

// Follows the chain inside a loop body from the accumulator iteration
// argument to the yield, counting macs; fails on any other consumer.
std::optional<int64_t> traceLoopBody(scf::ForOp forOp, unsigned position,
                                     llvm::SmallVectorImpl<Value> &values) {
  Value current = forOp.getRegionIterArg(position);
  values.push_back(current);
  int64_t macs = 0;
  while (Operation *user = soleUser(current)) {
    if (isFullQ15Mac(user) && user->getOperand(0) == current) {
      current = user->getResult(0);
      values.push_back(current);
      ++macs;
      continue;
    }
    if (user == forOp.getBody()->getTerminator() && user->getOperand(position) == current)
      return macs;
    return std::nullopt;
  }
  return std::nullopt;
}

// Traces acc_zero -> (macs | one static-trip-count loop)* -> acc_export and
// returns the web, or nothing when any step leaves the provable shape.
std::optional<AccumulatorWeb> traceWeb(AccZeroOp zero) {
  AccumulatorWeb web;
  Value current = zero.getAcc();
  web.values.push_back(current);
  while (Operation *user = soleUser(current)) {
    if (isFullQ15Mac(user) && user->getOperand(0) == current) {
      current = user->getResult(0);
      web.values.push_back(current);
      web.maxUpdates += 1;
      continue;
    }
    if (auto forOp = llvm::dyn_cast<scf::ForOp>(user)) {
      auto init = llvm::find(forOp.getInitArgs(), current);
      if (init == forOp.getInitArgs().end())
        return std::nullopt;
      unsigned position = std::distance(forOp.getInitArgs().begin(), init);
      std::optional<int64_t> lower = staticBound(forOp.getLowerBound());
      std::optional<int64_t> upper = staticBound(forOp.getUpperBound());
      std::optional<int64_t> step = staticBound(forOp.getStep());
      if (!lower || !upper || !step || *step <= 0)
        return std::nullopt;
      int64_t trips = *upper > *lower ? (*upper - *lower + *step - 1) / *step : 0;
      std::optional<int64_t> macs = traceLoopBody(forOp, position, web.values);
      if (!macs)
        return std::nullopt;
      web.maxUpdates += trips * *macs;
      current = forOp.getResult(position);
      web.values.push_back(current);
      continue;
    }
    if (llvm::isa<AccExportOp>(user))
      return web;
    return std::nullopt;
  }
  return std::nullopt;
}

struct WidenOndspExactAccumulators
    : ondrix::impl::WidenOndspExactAccumulatorsBase<WidenOndspExactAccumulators> {
  void runOnOperation() override {
    getOperation()->walk([&](AccZeroOp zero) {
      auto type = llvm::dyn_cast<AccType>(zero.getAcc().getType());
      if (!type || type.getLanes() != 1 || type.getFrac() != 30 ||
          type.getSignedness() != Signedness::Signed ||
          type.getUpdateOverflow() != OverflowMode::Wrap)
        return;
      auto storage = llvm::dyn_cast<IntegerType>(type.getStorage());
      if (!storage || storage.getWidth() > 40)
        return;
      std::optional<AccumulatorWeb> web = traceWeb(zero);
      if (!web)
        return;
      // Every partial sum is bounded by updates * 2^30 (full Q15 products),
      // so within this bound the declared wrap never fires and the target's
      // 40-bit saturation cannot fire either: both spell the exact sum.
      int64_t bound = int64_t(1) << (storage.getWidth() - 1);
      if (web->maxUpdates < 0 || web->maxUpdates > (bound - 1) >> 30)
        return;
      auto widened = AccType::get(type.getContext(), IntegerType::get(type.getContext(), 40),
                                  type.getFrac(), type.getSignedness(), OverflowMode::Saturate);
      for (Value value : web->values)
        value.setType(widened);
    });
  }
};

} // namespace

std::unique_ptr<mlir::Pass> ondrix::createWidenOndspExactAccumulatorsPass() {
  return std::make_unique<WidenOndspExactAccumulators>();
}
