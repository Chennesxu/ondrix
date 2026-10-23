#include "ondrix/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/Transforms/BufferViewFlowAnalysis.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "mlir/Pass/Pass.h"

namespace ondrix {
#define GEN_PASS_DEF_HOISTONDRIXREGIONBUFFERS
#include "ondrix/Transforms/Passes.h.inc"
} // namespace ondrix

using namespace mlir;

namespace {

// A user that only accesses the buffer, or forwards it to results the alias
// analysis tracks; a call or an address extraction may keep the pointer.
bool keepsAddress(Operation *user) {
  if (isRegionReturnLike(user))
    return false;
  return isa<memref::LoadOp, memref::StoreOp, memref::CopyOp, memref::DimOp, vector::LoadOp,
             vector::StoreOp, vector::TransferReadOp, vector::TransferWriteOp, ViewLikeOpInterface,
             RegionBranchOpInterface, arith::SelectOp>(user);
}

// Regions that run one execution at a time; a parallel iteration or an async
// task may overlap the next execution, so it needs a private buffer.
bool runsSequentially(Operation *op, Operation *function) {
  for (Operation *parent = op->getParentOp(); parent != function; parent = parent->getParentOp())
    if (!isa<scf::ForOp, scf::WhileOp, scf::IfOp>(parent))
      return false;
  return true;
}

class HoistOndrixRegionBuffersPass final
    : public ondrix::impl::HoistOndrixRegionBuffersBase<HoistOndrixRegionBuffersPass> {
public:
  void runOnOperation() override {
    func::FuncOp function = getOperation();
    if (function.isExternal())
      return;
    Block &entry = function.getBody().front();
    SmallVector<memref::AllocOp> nested;
    function.walk([&](memref::AllocOp alloc) {
      if (alloc->getParentOp() != function && alloc.getDynamicSizes().empty() &&
          alloc.getSymbolOperands().empty() && runsSequentially(alloc, function))
        nested.push_back(alloc);
    });
    BufferViewFlowAnalysis aliases(function);
    for (memref::AllocOp alloc : nested) {
      bool escapes = llvm::any_of(aliases.resolve(alloc.getMemref()), [&](Value alias) {
        return llvm::any_of(alias.getUsers(), [&](Operation *user) {
          return !keepsAddress(user) || !runsSequentially(user, function);
        });
      });
      if (!escapes)
        alloc->moveBefore(&entry, entry.begin());
    }
  }
};

} // namespace

std::unique_ptr<Pass> ondrix::createHoistOndrixRegionBuffersPass() {
  return std::make_unique<HoistOndrixRegionBuffersPass>();
}
