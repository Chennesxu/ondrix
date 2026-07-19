#include "ondrix/Dialect/ondrix/IR/OndrixAttrs.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

#include "ondrix/Dialect/ondrix/IR/OndrixDialect.h"

using namespace mlir;
using namespace ondrix::ir;

#include "ondrix/Dialect/ondrix/IR/OndrixEnums.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "ondrix/Dialect/ondrix/IR/OndrixAttrs.cpp.inc"

void OndrixDialect::registerAttributes() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "ondrix/Dialect/ondrix/IR/OndrixAttrs.cpp.inc"
      >();
}
