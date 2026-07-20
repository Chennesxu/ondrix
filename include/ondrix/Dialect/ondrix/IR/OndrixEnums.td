#ifndef ONDRIX_DIALECT_ONDRIX_IR_ONDRIXENUMS_TD
#define ONDRIX_DIALECT_ONDRIX_IR_ONDRIXENUMS_TD

include "mlir/IR/EnumAttr.td"

def Ondrix_FirBoundaryValid : I32EnumAttrCase<"Valid", 0, "valid">;
def Ondrix_FirBoundaryFull : I32EnumAttrCase<"Full", 1, "full">;
def Ondrix_FirBoundaryMode
    : I32EnumAttr<"FirBoundaryMode", "FIR output boundary mode",
                  [Ondrix_FirBoundaryValid, Ondrix_FirBoundaryFull]> {
  let cppNamespace = "::ondrix::ir";
  let genSpecializedAttr = 0;
}

#endif
