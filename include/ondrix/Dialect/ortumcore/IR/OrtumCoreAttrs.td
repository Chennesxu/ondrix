#ifndef ONDRIX_DIALECT_ORTUMCORE_IR_ORTUMCOREATTRS_TD
#define ONDRIX_DIALECT_ORTUMCORE_IR_ORTUMCOREATTRS_TD

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"
include "ondrix/Dialect/ortumcore/IR/OrtumCoreDialect.td"
include "ondrix/Dialect/ortumcore/IR/OrtumCoreEnums.td"

def OrtumCore_FftStageKindAttr
    : EnumAttr<OrtumCore_Dialect, OrtumCore_FftStageKind, "fft_stage_kind">;

#endif
