#ifndef ONDRIX_DIALECT_ORTUMCORE_IR_ORTUMCORETYPES_TD
#define ONDRIX_DIALECT_ORTUMCORE_IR_ORTUMCORETYPES_TD

include "mlir/IR/AttrTypeBase.td"
include "ondrix/Dialect/ortumcore/IR/OrtumCoreDialect.td"

class OrtumCore_Type<string name, string typeMnemonic>
    : TypeDef<OrtumCore_Dialect, name> {
  let mnemonic = typeMnemonic;
}

def OrtumCore_AccumType : OrtumCore_Type<"Accum", "acc"> {
  let summary = "Target accumulator semantic value";
}

def OrtumCore_AccumPairType : OrtumCore_Type<"AccumPair", "acc_pair"> {
  let summary = "Pair of target accumulator semantic values";
}

def OrtumCore_VecType : OrtumCore_Type<"Vec", "vec"> {
  let summary = "Packed vector semantic value";
}

def OrtumCore_VecPairType : OrtumCore_Type<"VecPair", "vec_pair"> {
  let summary = "Pair of packed vector semantic values";
}

def OrtumCore_VecStateType : OrtumCore_Type<"VecState", "vec.state"> {
  let summary = "Explicit packed-vector mode/state token";
}

#endif
