#ifndef ONDRIX_DIALECT_ONDRIX_IR_ONDRIXATTRS_H
#define ONDRIX_DIALECT_ONDRIX_IR_ONDRIXATTRS_H

#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"

#include "ondrix/Dialect/ondrix/IR/OndrixEnums.h"

#define GET_ATTRDEF_CLASSES
#include "ondrix/Dialect/ondrix/IR/OndrixAttrs.h.inc"

#endif
