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

/// Position of the first LLVM argument that `argument` expands to under the
/// default memref descriptor convention: a ranked memref becomes two pointers,
/// an offset and a size and a stride per dimension; anything else stays one.
int64_t getExpandedArgumentPosition(func::FuncOp function, unsigned argument) {
  int64_t position = 0;
  for (unsigned index = 0; index < argument; ++index) {
    Type type = function.getArgument(index).getType();
    if (auto memref = dyn_cast<MemRefType>(type))
      position += 3 + 2 * memref.getRank();
    else if (isa<UnrankedMemRefType>(type))
      position += 2;
    else
      position += 1;
  }
  return position;
}

/// Records that the forwarded destination's allocated and aligned pointers may
/// carry `noalias` once the descriptor is expanded: the parameter stands for a
/// result the function used to allocate itself, which aliased nothing.
void markDestinationNoAlias(func::FuncOp function, BlockArgument destination) {
  int64_t first = getExpandedArgumentPosition(function, destination.getArgNumber());
  SmallVector<int64_t> positions;
  if (auto existing =
          function->getAttrOfType<DenseI64ArrayAttr>(ondrix::kNoAliasPointerArgumentsAttr))
    positions.assign(existing.asArrayRef().begin(), existing.asArrayRef().end());
  positions.push_back(first);
  positions.push_back(first + 1);
  function->setAttr(ondrix::kNoAliasPointerArgumentsAttr,
                    DenseI64ArrayAttr::get(function.getContext(), positions));
}

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
        markDestinationNoAlias(function, cast<BlockArgument>(destination));
      }
    });
  }
};

} // namespace

std::unique_ptr<Pass> ondrix::createForwardOndrixResultBuffersPass() {
  return std::make_unique<ForwardOndrixResultBuffersPass>();
}
