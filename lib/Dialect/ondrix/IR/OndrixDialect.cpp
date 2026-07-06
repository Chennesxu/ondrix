#include "ondrix/Dialect/ondrix/IR/OndrixDialect.h"

#include "ondrix/Dialect/ondrix/IR/OndrixOps.h"

using namespace mlir;
using namespace ondrix::ir;

#include "ondrix/Dialect/ondrix/IR/OndrixOpsDialect.cpp.inc"

void OndrixDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "ondrix/Dialect/ondrix/IR/OndrixOps.cpp.inc"
      >();
}
