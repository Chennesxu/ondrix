#include "ondrix/Dialect/ondsp/IR/OndspDialect.h"

#include "ondrix/Dialect/ondsp/IR/OndspAttrs.h"
#include "ondrix/Dialect/ondsp/IR/OndspOps.h"
#include "ondrix/Dialect/ondsp/IR/OndspTypes.h"

using namespace mlir;
using namespace ondrix::ondsp;

#include "ondrix/Dialect/ondsp/IR/OndspOpsDialect.cpp.inc"

void OndspDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "ondrix/Dialect/ondsp/IR/OndspOps.cpp.inc"
      >();
  registerAttributes();
  registerTypes();
}
