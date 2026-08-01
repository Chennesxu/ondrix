#include "ondrix/Conversion/Utils/MemRefLayoutUtils.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/FunctionInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"

using namespace mlir;

namespace ondrix::conversion {

bool hasDefaultLLVMVectorMemorySpace(MemRefType type) {
  Attribute memorySpace = type.getMemorySpace();
  if (!memorySpace)
    return true;
  auto integerSpace = dyn_cast<IntegerAttr>(memorySpace);
  return integerSpace && integerSpace.getValue().isZero();
}

Value resolveMemRefBase(Value value) {
  while (auto view = value.getDefiningOp<ViewLikeOpInterface>())
    value = view.getViewSource();
  return value;
}

/// Whether a base value is one whose storage this analysis can reason about at
/// all. Anything else — an `arith.select` between two memrefs, an `scf.if`
/// result, a call result — is an opaque producer that may hand back storage
/// already named by another operand, and is therefore not a fresh base.
///
/// Among block arguments, only a function ENTRY argument qualifies: its
/// storage is chosen by the caller, so disjointness against another entry
/// argument is a fact the caller owns. Every other block argument — a branch
/// successor argument, a loop's iter_args — merely renames whatever its
/// in-function predecessors pass, so the very same buffer and a view of it can
/// arrive as two distinct arguments with no caller anywhere to discharge the
/// overlap. Those are opaque.
static bool hasKnownProvenance(Value base) {
  if (auto argument = dyn_cast<BlockArgument>(base)) {
    Block *owner = argument.getOwner();
    return owner->isEntryBlock() && isa<FunctionOpInterface>(owner->getParentOp());
  }
  Operation *producer = base.getDefiningOp();
  return producer && isa<memref::AllocOp, memref::AllocaOp, memref::GetGlobalOp>(producer);
}

bool mayShareStorage(Value lhs, Value rhs) {
  Value lhsBase = resolveMemRefBase(lhs);
  Value rhsBase = resolveMemRefBase(rhs);
  if (lhsBase == rhsBase)
    return true;
  // Fail closed on anything this analysis cannot place. An opaque producer can
  // return storage another operand already names, and answering "distinct"
  // there would silently authorize a rewrite on a false premise.
  if (!hasKnownProvenance(lhsBase) || !hasKnownProvenance(rhsBase))
    return true;
  // Two reads of one global are distinct SSA values naming one buffer.
  auto lhsGlobal = lhsBase.getDefiningOp<memref::GetGlobalOp>();
  auto rhsGlobal = rhsBase.getDefiningOp<memref::GetGlobalOp>();
  if (lhsGlobal && rhsGlobal)
    return lhsGlobal.getName() == rhsGlobal.getName();
  // What remains is a pair of distinct known bases. An allocation is fresh
  // storage, so it shares with nothing else. Two distinct function entry
  // arguments, or an entry argument against a global, are not decidable here:
  // they are precisely the runtime residual a caller owns, not a fact this
  // returns.
  return false;
}

} // namespace ondrix::conversion
