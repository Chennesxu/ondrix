#ifndef ONDRIX_DIALECT_ONDRIX_IR_ONDRIXATTRS_TD
#define ONDRIX_DIALECT_ONDRIX_IR_ONDRIXATTRS_TD

include "mlir/IR/AttrTypeBase.td"
include "ondrix/Dialect/ondrix/IR/OndrixDialect.td"
include "ondrix/Dialect/ondrix/IR/OndrixEnums.td"

def Ondrix_FirBoundaryModeAttr
    : EnumAttr<Ondrix_Dialect, Ondrix_FirBoundaryMode, "fir_boundary"> {
  let assemblyFormat = "`<` $value `>`";
}

#endif
