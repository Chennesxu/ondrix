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
  let summary = "Target accumulator pair semantic value";
  let description = [{
    Models a target semantic value carried by paired target storage. Physical
    register allocation and pair constraints remain backend-private.
  }];
}

def OrtumCore_FpType : OrtumCore_Type<"Fp", "fp"> {
  let summary = "Target floating-point semantic value";
}

def OrtumCore_FpPairType : OrtumCore_Type<"FpPair", "fp_pair"> {
  let summary = "Target floating-point pair semantic value";
  let description = [{
    Models a target floating-point semantic value carried by paired target
    storage. Physical register constraints are handled by the backend.
  }];
}

def OrtumCore_VecType : OrtumCore_Type<"Vec", "vec"> {
  let summary = "Packed vector semantic value";
}

def OrtumCore_VecPairType : OrtumCore_Type<"VecPair", "vec_pair"> {
  let summary = "Pair of packed vector semantic values";
}

def OrtumCore_VecStateType : OrtumCore_Type<"VecState", "vec.state"> {
  let summary = "Experimental packed-vector mode/state token";
  let description = [{
    Ordering token for packed-vector mode changes and dependent packed-vector
    operations. It is target semantic state, not a public physical register.
  }];
}

#endif
