#ifndef ONDRIX_DIALECT_ONDSP_IR_ONDSPDIALECT_TD
#define ONDRIX_DIALECT_ONDSP_IR_ONDSPDIALECT_TD

include "mlir/IR/OpBase.td"

def Ondsp_Dialect : Dialect {
  let name = "ondsp";
  let summary = "Target-independent DSP numeric semantic dialect";
  let cppNamespace = "::ondrix::ondsp";

  let useDefaultAttributePrinterParser = 1;
  let useDefaultTypePrinterParser = 1;

  let extraClassDeclaration = [{
    void registerAttributes();
    void registerTypes();
  }];
}

class Ondsp_Op<string mnemonic, list<Trait> traits = []>
    : Op<Ondsp_Dialect, mnemonic, traits>;

#endif
