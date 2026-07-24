#ifndef ONDRIX_DIALECT_ONDRIX_IR_ONDRIXATTRS_TD
#define ONDRIX_DIALECT_ONDRIX_IR_ONDRIXATTRS_TD

include "mlir/IR/AttrTypeBase.td"
include "ondrix/Dialect/ondrix/IR/OndrixDialect.td"
include "ondrix/Dialect/ondrix/IR/OndrixEnums.td"

def Ondrix_FirBoundaryModeAttr
    : EnumAttr<Ondrix_Dialect, Ondrix_FirBoundaryMode, "fir_boundary"> {
  let assemblyFormat = "`<` $value `>`";
}

def Ondrix_Conv1DModeAttr
    : EnumAttr<Ondrix_Dialect, Ondrix_Conv1DMode, "conv1d_mode"> {
  let assemblyFormat = "`<` $value `>`";
}

def Ondrix_CfftDirectionAttr
    : EnumAttr<Ondrix_Dialect, Ondrix_CfftDirection, "cfft_direction"> {
  let assemblyFormat = "`<` $value `>`";
}

#endif
