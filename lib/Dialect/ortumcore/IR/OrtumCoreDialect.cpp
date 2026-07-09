#include "ondrix/Dialect/ortumcore/IR/OrtumCoreDialect.h"

#include "ondrix/Dialect/ortumcore/IR/OrtumCoreOps.h"
#include "ondrix/Dialect/ortumcore/IR/OrtumCoreTypes.h"

using namespace mlir;
using namespace ondrix::ortumcore;

#include "ondrix/Dialect/ortumcore/IR/OrtumCoreOpsDialect.cpp.inc"

void OrtumCoreDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "ondrix/Dialect/ortumcore/IR/OrtumCoreOps.cpp.inc"
      >();
  registerAttributes();
  registerTypes();
}
