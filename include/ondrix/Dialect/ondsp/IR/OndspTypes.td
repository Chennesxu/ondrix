#ifndef ONDRIX_DIALECT_ONDSP_IR_ONDSPTYPES_TD
#define ONDRIX_DIALECT_ONDSP_IR_ONDSPTYPES_TD

include "mlir/IR/AttrTypeBase.td"
include "ondrix/Dialect/ondsp/IR/OndspDialect.td"
include "ondrix/Dialect/ondsp/IR/OndspEnums.td"

class Ondsp_Type<string name, string typeMnemonic>
    : TypeDef<Ondsp_Dialect, name> {
  let mnemonic = typeMnemonic;
}

def Ondsp_AccType : Ondsp_Type<"Acc", "acc"> {
  let summary = "DSP accumulator state with explicit storage precision";
  let parameters = (ins
    "mlir::Type":$storage,
    "unsigned":$frac,
    "Signedness":$signedness
  );
  let assemblyFormat =
      "`<` `storage` `=` $storage `,` `frac` `=` $frac `,` $signedness `>`";
  let genVerifyDecl = 1;
}

def Ondsp_Acc : Type<
    CPred<"::llvm::isa<::ondrix::ondsp::AccType>($_self)">,
    "ondsp accumulator type", "::ondrix::ondsp::AccType">;

#endif
