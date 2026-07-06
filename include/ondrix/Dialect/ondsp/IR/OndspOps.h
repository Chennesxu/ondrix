#ifndef ONDRIX_DIALECT_ONDSP_IR_ONDSPOPS_H
#define ONDRIX_DIALECT_ONDSP_IR_ONDSPOPS_H

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "ondrix/Dialect/ondsp/IR/OndspAttrs.h"
#include "ondrix/Dialect/ondsp/IR/OndspTypes.h"

#define GET_OP_CLASSES
#include "ondrix/Dialect/ondsp/IR/OndspOps.h.inc"

#endif
