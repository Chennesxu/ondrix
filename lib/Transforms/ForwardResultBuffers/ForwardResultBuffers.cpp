#include "ondrix/Transforms/Passes.h"

#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "forward-ondrix-result-buffers"

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

/// Whether `user` is one of the plain accesses that read or write the buffer
/// in place without letting its address or a view of it escape.
bool isPlainAccess(Operation *user) {
  return isa<memref::LoadOp, memref::StoreOp, memref::CopyOp, memref::DeallocOp, vector::LoadOp,
             vector::StoreOp, vector::TransferReadOp, vector::TransferWriteOp>(user);
}

/// The buffer a memref value was derived from, through view-like operations.
Value getRootBuffer(Value memref) {
  while (auto view = memref.getDefiningOp<ViewLikeOpInterface>())
    memref = view.getViewSource();
  return memref;
}

/// Whether every memory access in `function` roots at one of its arguments, a
/// local allocation, or a constant global: the storage set the declared
/// calling convention speaks about. A call, an effect on unknown storage, or a
/// mutable global reaches storage the destination could alias.
bool accessesOnlyDeclaredStorage(func::FuncOp function) {
  Block &entry = function.getBody().front();
  auto isDeclaredRoot = [&](Value root) {
    if (auto argument = dyn_cast<BlockArgument>(root))
      return argument.getOwner() == &entry;
    Operation *definition = root.getDefiningOp();
    if (isa_and_nonnull<memref::AllocOp, memref::AllocaOp>(definition))
      return true;
    if (auto global = dyn_cast_or_null<memref::GetGlobalOp>(definition)) {
      auto symbol =
          SymbolTable::lookupNearestSymbolFrom<memref::GlobalOp>(global, global.getNameAttr());
      return symbol && symbol.getConstant();
    }
    return false;
  };
  auto refuse = [&](Operation *operation, StringRef why) {
    LLVM_DEBUG(llvm::dbgs() << "refusing " << function.getName() << ": " << why << " at "
                            << *operation << "\n");
    return WalkResult::interrupt();
  };
  WalkResult result = function.walk([&](Operation *operation) {
    if (operation == function.getOperation())
      return WalkResult::advance();
    if (isa<CallOpInterface>(operation))
      return refuse(operation, "call");
    // A runtime assertion carries no effect declaration but touches no
    // storage; the lowering's equal-length checks are this shape.
    if (isMemoryEffectFree(operation) || isa<cf::AssertOp>(operation))
      return WalkResult::advance();
    auto effects = dyn_cast<MemoryEffectOpInterface>(operation);
    if (!effects)
      return operation->hasTrait<OpTrait::HasRecursiveMemoryEffects>()
                 ? WalkResult::advance()
                 : refuse(operation, "no memory-effect information");
    SmallVector<MemoryEffects::EffectInstance> instances;
    effects.getEffects(instances);
    for (const MemoryEffects::EffectInstance &instance : instances) {
      if (isa<MemoryEffects::Allocate, MemoryEffects::Free>(instance.getEffect()))
        continue;
      Value value = instance.getValue();
      if (!value || !isa<MemRefType, UnrankedMemRefType>(value.getType()) ||
          !isDeclaredRoot(getRootBuffer(value)))
        return refuse(operation, "effect on storage the convention does not name");
    }
    return WalkResult::advance();
  });
  return !result.wasInterrupted();
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
    if (!isPlainAccess(user))
      return false;
    Operation *ancestor = copy->getBlock()->findAncestorOpInBlock(*user);
    if (!ancestor || !ancestor->isBeforeInBlock(copy))
      return false;
  }
  return true;
}

class ForwardOndrixResultBuffersPass final
    : public ondrix::impl::ForwardOndrixResultBuffersBase<ForwardOndrixResultBuffersPass> {
public:
  using ondrix::impl::ForwardOndrixResultBuffersBase<
      ForwardOndrixResultBuffersPass>::ForwardOndrixResultBuffersBase;

  void runOnOperation() override {
    // The rewrite reads inputs after writing the destination, which is only
    // the original program when the caller keeps them apart; that is a
    // calling convention the pipeline declares, never something inferred.
    if (!distinctOutParams)
      return;
    getOperation().walk([](func::FuncOp function) {
      if (function.isExternal() || !accessesOnlyDeclaredStorage(function))
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

std::unique_ptr<Pass>
ondrix::createForwardOndrixResultBuffersPass(const ForwardOndrixResultBuffersOptions &options) {
  return std::make_unique<ForwardOndrixResultBuffersPass>(options);
}
