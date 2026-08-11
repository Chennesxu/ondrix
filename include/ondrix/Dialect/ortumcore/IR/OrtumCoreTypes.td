#ifndef ONDRIX_DIALECT_ORTUMCORE_IR_ORTUMCORETYPES_TD
#define ONDRIX_DIALECT_ORTUMCORE_IR_ORTUMCORETYPES_TD

include "mlir/IR/AttrTypeBase.td"
include "ondrix/Dialect/ortumcore/IR/OrtumCoreDialect.td"

class OrtumCore_Type<string name, string typeMnemonic>
    : TypeDef<OrtumCore_Dialect, name> {
  let mnemonic = typeMnemonic;
}

def OrtumCore_AccumType : OrtumCore_Type<"Accum", "acc"> {
  let summary = "Signed i40 frac30 saturating accumulator semantic value";
  let description = [{
    Represents signed 40-bit two's-complement raw accumulator state with 30
    fractional bits. Operations that update this value must preserve this
    domain. The type models an abstract target capability and does not expose
    physical storage or register-allocation constraints.
  }];
}

#endif
