#include "ondrix/Dialect/ortumcore/IR/OrtumCoreAttrs.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

#include "ondrix/Dialect/ortumcore/IR/OrtumCoreDialect.h"

using namespace mlir;
using namespace ondrix::ortumcore;

#include "ondrix/Dialect/ortumcore/IR/OrtumCoreEnums.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "ondrix/Dialect/ortumcore/IR/OrtumCoreAttrs.cpp.inc"

void OrtumCoreDialect::registerAttributes() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "ondrix/Dialect/ortumcore/IR/OrtumCoreAttrs.cpp.inc"
      >();
}
