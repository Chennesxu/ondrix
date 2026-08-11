#ifndef ONDRIX_DIALECT_ORTUMCORE_IR_ORTUMCOREATTRS_TD
#define ONDRIX_DIALECT_ORTUMCORE_IR_ORTUMCOREATTRS_TD

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"
include "ondrix/Dialect/ortumcore/IR/OrtumCoreDialect.td"
include "ondrix/Dialect/ortumcore/IR/OrtumCoreEnums.td"

def OrtumCore_CxRoundingAttr
    : EnumAttr<OrtumCore_Dialect, OrtumCore_CxRounding, "cx_rounding">;
def OrtumCore_CxOverflowAttr
    : EnumAttr<OrtumCore_Dialect, OrtumCore_CxOverflow, "cx_overflow">;
def OrtumCore_CxLayoutAttr
    : EnumAttr<OrtumCore_Dialect, OrtumCore_CxLayout, "cx_layout">;
def OrtumCore_CxBflyVariantAttr
    : EnumAttr<OrtumCore_Dialect, OrtumCore_CxBflyVariant, "cx_bfly_variant">;

#endif
