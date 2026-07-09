#include "ondrix/Dialect/ortumcore/IR/OrtumCoreOps.h"

#include "mlir/IR/Diagnostics.h"

using namespace mlir;
using namespace ondrix::ortumcore;

#define GET_OP_CLASSES
#include "ondrix/Dialect/ortumcore/IR/OrtumCoreOps.cpp.inc"
