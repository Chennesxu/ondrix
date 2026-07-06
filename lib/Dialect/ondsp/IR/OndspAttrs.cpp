#include "ondrix/Dialect/ondsp/IR/OndspAttrs.h"

#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

#include "ondrix/Dialect/ondsp/IR/OndspDialect.h"

using namespace mlir;
using namespace ondrix::ondsp;

#include "ondrix/Dialect/ondsp/IR/OndspEnums.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "ondrix/Dialect/ondsp/IR/OndspAttrs.cpp.inc"

void OndspDialect::registerAttributes() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "ondrix/Dialect/ondsp/IR/OndspAttrs.cpp.inc"
      >();
}

LogicalResult FixedAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                                Signedness signedness, Type storage,
                                unsigned frac) {
  (void)signedness;
  auto intType = storage.dyn_cast<IntegerType>();
  if (!intType)
    return emitError() << "fixed numeric storage must be an integer type";

  unsigned width = intType.getWidth();
  if (width == 0)
    return emitError() << "fixed numeric storage must have non-zero width";

  if (frac > width)
    return emitError() << "fixed numeric frac must not exceed storage width";

  return success();
}
