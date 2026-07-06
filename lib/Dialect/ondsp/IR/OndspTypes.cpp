#include "ondrix/Dialect/ondsp/IR/OndspTypes.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

#include "ondrix/Dialect/ondsp/IR/OndspDialect.h"

using namespace mlir;
using namespace ondrix::ondsp;

#define GET_TYPEDEF_CLASSES
#include "ondrix/Dialect/ondsp/IR/OndspTypes.cpp.inc"

void OndspDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "ondrix/Dialect/ondsp/IR/OndspTypes.cpp.inc"
      >();
}

LogicalResult AccType::verify(function_ref<InFlightDiagnostic()> emitError, Type storage,
                              unsigned frac, Signedness signedness) {
  (void)signedness;
  auto intType = storage.dyn_cast<IntegerType>();
  if (!intType)
    return emitError() << "accumulator storage must be an integer type";

  unsigned width = intType.getWidth();
  if (width < 32)
    return emitError() << "accumulator storage must be at least 32 bits";

  if (frac > width)
    return emitError() << "accumulator frac must not exceed storage width";

  return success();
}
