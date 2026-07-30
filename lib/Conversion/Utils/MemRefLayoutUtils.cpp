#include "ondrix/Conversion/Utils/MemRefLayoutUtils.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
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

bool mayShareStorage(Value lhs, Value rhs) {
  Value lhsBase = resolveMemRefBase(lhs);
  Value rhsBase = resolveMemRefBase(rhs);
  if (lhsBase == rhsBase)
    return true;
  // Two reads of one global are distinct SSA values naming one buffer.
  auto lhsGlobal = lhsBase.getDefiningOp<memref::GetGlobalOp>();
  auto rhsGlobal = rhsBase.getDefiningOp<memref::GetGlobalOp>();
  return lhsGlobal && rhsGlobal && lhsGlobal.getName() == rhsGlobal.getName();
}

} // namespace ondrix::conversion
