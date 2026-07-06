#ifndef ONDRIX_DIALECT_ONDRIX_IR_ONDRIXDIALECT_TD
#define ONDRIX_DIALECT_ONDRIX_IR_ONDRIXDIALECT_TD

include "mlir/IR/OpBase.td"

def Ondrix_Dialect : Dialect {
  let name = "ondrix";
  let summary = "High-level DSP kernel intent dialect";
  let cppNamespace = "::ondrix::ir";
}

class Ondrix_Op<string mnemonic, list<Trait> traits = []>
    : Op<Ondrix_Dialect, mnemonic, traits>;

#endif
