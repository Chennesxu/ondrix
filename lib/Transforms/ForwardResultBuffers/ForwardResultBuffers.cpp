#include "ondrix/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace ondrix {
#define GEN_PASS_DEF_FORWARDONDRIXRESULTBUFFERS
#include "ondrix/Transforms/Passes.h.inc"
} // namespace ondrix

using namespace mlir;

namespace {

/// Whether `copy` is the whole-buffer hand-off of a local allocation into an
/// otherwise untouched out-parameter, in the function's entry block, with no
/// use of the allocation after it except its deallocation.
bool isForwardableResultCopy(memref::CopyOp copy) {
  auto destination = dyn_cast<BlockArgument>(copy.getTarget());
  if (!destination || !destination.hasOneUse() ||
      !isa<func::FuncOp>(destination.getOwner()->getParentOp()))
    return false;
  Value source = copy.getSource();
  if (source.getType() != destination.getType())
    return false;
  auto type = cast<MemRefType>(source.getType());
  if (!type.hasStaticShape() || !type.getLayout().isIdentity())
    return false;
  Operation *allocation = source.getDefiningOp();
  if (!isa_and_nonnull<memref::AllocOp, memref::AllocaOp>(allocation) ||
      allocation->getBlock() != copy->getBlock())
    return false;
  // Every write into the allocation must be complete when the copy runs, so
  // nothing may use it afterwards; a deallocation is the one exception.
  for (Operation *user : source.getUsers()) {
    if (user == copy || isa<memref::DeallocOp>(user))
      continue;
    Operation *ancestor = copy->getBlock()->findAncestorOpInBlock(*user);
    if (!ancestor || !ancestor->isBeforeInBlock(copy))
      return false;
  }
  return true;
}

class ForwardOndrixResultBuffersPass final
    : public ondrix::impl::ForwardOndrixResultBuffersBase<ForwardOndrixResultBuffersPass> {
public:
  void runOnOperation() override {
    getOperation().walk([](func::FuncOp function) {
      if (function.isExternal())
        return;
      SmallVector<memref::CopyOp> copies;
      for (Operation &operation : function.getBody().front())
        if (auto copy = dyn_cast<memref::CopyOp>(operation))
          if (isForwardableResultCopy(copy))
            copies.push_back(copy);
      for (memref::CopyOp copy : copies) {
        Value source = copy.getSource();
        Value destination = copy.getTarget();
        SmallVector<Operation *> deallocations;
        for (Operation *user : source.getUsers())
          if (isa<memref::DeallocOp>(user))
            deallocations.push_back(user);
        for (Operation *deallocation : deallocations)
          deallocation->erase();
        copy.erase();
        Operation *allocation = source.getDefiningOp();
        source.replaceAllUsesWith(destination);
        allocation->erase();
      }
    });
  }
};

} // namespace

std::unique_ptr<Pass> ondrix::createForwardOndrixResultBuffersPass() {
  return std::make_unique<ForwardOndrixResultBuffersPass>();
}
