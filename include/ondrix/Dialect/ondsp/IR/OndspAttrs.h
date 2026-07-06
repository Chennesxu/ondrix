#ifndef ONDRIX_DIALECT_ONDSP_IR_ONDSPATTRS_H
#define ONDRIX_DIALECT_ONDSP_IR_ONDSPATTRS_H

#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"

#include "ondrix/Dialect/ondsp/IR/OndspEnums.h"

#define GET_ATTRDEF_CLASSES
#include "ondrix/Dialect/ondsp/IR/OndspAttrs.h.inc"

#endif
