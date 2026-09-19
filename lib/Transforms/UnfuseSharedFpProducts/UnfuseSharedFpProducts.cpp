#include "ondrix/Transforms/Passes.h"

#include "ondrix/Dialect/ondsp/IR/OndspDialect.h"
#include "ondrix/Dialect/ondsp/IR/OndspSemantics.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace ondrix {
#define GEN_PASS_DEF_UNFUSEONDSPSHAREDFPPRODUCTS
#include "ondrix/Transforms/Passes.h.inc"
} // namespace ondrix

using namespace mlir;

namespace {

/// A multiply that computes `lhs * rhs` for some other reader. It may sit
/// after `op`: its operands are `op`'s own, so they already dominate `op` and
/// the multiply can be moved up to meet it. Only the same block is considered,
/// which keeps that move a reordering inside one list. Commutativity is a
/// term-matching convenience -- the operands are the same two values.
arith::MulFOp findSharedProduct(Operation *op, Value lhs, Value rhs) {
  for (Operation *user : lhs.getUsers()) {
    auto multiply = dyn_cast<arith::MulFOp>(user);
    if (!multiply || multiply.getType() != op->getResult(0).getType() ||
        multiply->getBlock() != op->getBlock())
      continue;
    Value a = multiply.getLhs(), b = multiply.getRhs();
    if ((a == lhs && b == rhs) || (a == rhs && b == lhs))
      return multiply;
  }
  return {};
}

class UnfuseOndspSharedFpProductsPass final
    : public ondrix::impl::UnfuseOndspSharedFpProductsBase<UnfuseOndspSharedFpProductsPass> {
public:
  using ondrix::impl::UnfuseOndspSharedFpProductsBase<
      UnfuseOndspSharedFpProductsPass>::UnfuseOndspSharedFpProductsBase;

  void runOnOperation() override {
    SmallVector<std::pair<math::FmaOp, arith::MulFOp>> candidates;
    getOperation().walk([&](math::FmaOp op) {
      if (!ondrix::ondsp::hasSpentFastPermission(op,
                                                 ondrix::ondsp::FastPermission::FuseMultiplyAdd))
        return;
      if (arith::MulFOp shared = findSharedProduct(op, op.getA(), op.getB()))
        candidates.emplace_back(op, shared);
    });
    for (auto [op, shared] : candidates) {
      // The multiply's operands are the fused operation's own, so they already
      // dominate it and a later multiply only has to move up to meet it.
      if (op->isBeforeInBlock(shared.getOperation()))
        shared->moveBefore(op);
      OpBuilder builder(op);
      // The ordered form this returns to is the one `off` builds: the
      // accumulator on the left, the product it already had on the right.
      Value sum = builder.create<arith::AddFOp>(op.getLoc(), op.getC(), shared.getResult());
      op.getResult().replaceAllUsesWith(sum);
      op.erase();
    }
  }
};

} // namespace

std::unique_ptr<Pass> ondrix::createUnfuseOndspSharedFpProductsPass() {
  return std::make_unique<UnfuseOndspSharedFpProductsPass>();
}
