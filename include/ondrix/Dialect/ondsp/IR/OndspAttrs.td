#ifndef ONDRIX_DIALECT_ONDSP_IR_ONDSPATTRS_TD
#define ONDRIX_DIALECT_ONDSP_IR_ONDSPATTRS_TD

include "mlir/IR/AttrTypeBase.td"
include "ondrix/Dialect/ondsp/IR/OndspDialect.td"
include "ondrix/Dialect/ondsp/IR/OndspEnums.td"

class Ondsp_Attr<string name, string attrMnemonic>
    : AttrDef<Ondsp_Dialect, name> {
  let mnemonic = attrMnemonic;
}

def Ondsp_FixedAttr : Ondsp_Attr<"Fixed", "fixed"> {
  let summary = "Fixed-point numeric interpretation for an integer storage type";
  let parameters = (ins
    "Signedness":$signedness,
    "mlir::Type":$storage,
    "unsigned":$frac
  );
  let assemblyFormat =
      "`<` $signedness `,` `storage` `=` $storage `,` `frac` `=` $frac `>`";
  let genVerifyDecl = 1;
}

def Ondsp_FpAttr : Ondsp_Attr<"Fp", "fp"> {
  let summary = "Floating-point numeric interpretation";
  let parameters = (ins
    "mlir::Type":$format,
    "FpContractMode":$contract
  );
  let assemblyFormat =
      "`<` `format` `=` $format `,` `contract` `=` $contract `>`";
}

def Ondsp_CxLayoutAttr : Ondsp_Attr<"CxLayout", "cx_layout"> {
  let summary = "Complex value storage layout";
  let parameters = (ins "ComplexLayout":$layout);
  let assemblyFormat = "`<` $layout `>`";
}

def Ondsp_ScaleAttr : Ondsp_Attr<"Scale", "scale"> {
  let summary = "Explicit fixed-point scaling policy";
  let parameters = (ins
    "unsigned":$preShiftLeft,
    "unsigned":$postShiftRight,
    "RoundingMode":$rounding,
    "OverflowMode":$overflow,
    "mlir::Type":$saturateTo
  );
  let genVerifyDecl = 1;
  let assemblyFormat =
      "`<` `pre_shift_left` `=` $preShiftLeft `,` `post_shift_right` `=` $postShiftRight `,` `rounding` `=` $rounding `,` `overflow` `=` $overflow `,` `saturate_to` `=` $saturateTo `>`";
}

#endif
