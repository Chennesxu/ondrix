#ifndef ONDRIX_DIALECT_ONDRIX_IR_ONDRIXDIALECT_TD
#define ONDRIX_DIALECT_ONDRIX_IR_ONDRIXDIALECT_TD

include "mlir/IR/OpBase.td"

def Ondrix_Dialect : Dialect {
  let name = "ondrix";
  let summary = "High-level DSP kernel intent dialect";
  let cppNamespace = "::ondrix::ir";

  let useDefaultAttributePrinterParser = 1;

  let extraClassDeclaration = [{
    void registerAttributes();
  }];
}

class Ondrix_Op<string mnemonic, list<Trait> traits = []>
    : Op<Ondrix_Dialect, mnemonic, traits>;

#endif
