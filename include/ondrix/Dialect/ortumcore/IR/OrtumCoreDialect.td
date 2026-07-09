#ifndef ONDRIX_DIALECT_ORTUMCORE_IR_ORTUMCOREDIALECT_TD
#define ONDRIX_DIALECT_ORTUMCORE_IR_ORTUMCOREDIALECT_TD

include "mlir/IR/OpBase.td"

def OrtumCore_Dialect : Dialect {
  let name = "ortumcore";
  let summary = "Target-specific DSP hardware semantic dialect";
  let cppNamespace = "::ondrix::ortumcore";

  let useDefaultAttributePrinterParser = 1;
  let useDefaultTypePrinterParser = 1;

  let extraClassDeclaration = [{
    void registerAttributes();
    void registerTypes();
  }];
}

class OrtumCore_Op<string mnemonic, list<Trait> traits = []>
    : Op<OrtumCore_Dialect, mnemonic, traits>;

#endif
