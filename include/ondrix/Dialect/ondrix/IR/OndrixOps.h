#ifndef ONDRIX_DIALECT_ONDRIX_IR_ONDRIXOPS_H
#define ONDRIX_DIALECT_ONDRIX_IR_ONDRIXOPS_H

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#define GET_OP_CLASSES
#include "ondrix/Dialect/ondrix/IR/OndrixOps.h.inc"

#endif
