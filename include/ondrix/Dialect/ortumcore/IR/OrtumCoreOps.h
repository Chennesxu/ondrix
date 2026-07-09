#ifndef ONDRIX_DIALECT_ORTUMCORE_IR_ORTUMCOREOPS_H
#define ONDRIX_DIALECT_ORTUMCORE_IR_ORTUMCOREOPS_H

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "ondrix/Dialect/ortumcore/IR/OrtumCoreAttrs.h"
#include "ondrix/Dialect/ortumcore/IR/OrtumCoreTypes.h"

#define GET_OP_CLASSES
#include "ondrix/Dialect/ortumcore/IR/OrtumCoreOps.h.inc"

#endif
