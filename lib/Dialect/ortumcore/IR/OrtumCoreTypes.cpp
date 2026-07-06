#include "ondrix/Dialect/ortumcore/IR/OrtumCoreTypes.h"

#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

#include "ondrix/Dialect/ortumcore/IR/OrtumCoreDialect.h"

using namespace mlir;
using namespace ondrix::ortumcore;

#define GET_TYPEDEF_CLASSES
#include "ondrix/Dialect/ortumcore/IR/OrtumCoreTypes.cpp.inc"

void OrtumCoreDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "ondrix/Dialect/ortumcore/IR/OrtumCoreTypes.cpp.inc"
      >();
}
