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

def Ondrix_Conv1DConvolution
    : I32EnumAttrCase<"Convolution", 0, "convolution">;
def Ondrix_Conv1DCorrelation
    : I32EnumAttrCase<"Correlation", 1, "correlation">;
def Ondrix_Conv1DMode
    : I32EnumAttr<"Conv1DMode", "one-dimensional convolution family mode",
                  [Ondrix_Conv1DConvolution, Ondrix_Conv1DCorrelation]> {
  let cppNamespace = "::ondrix::ir";
  let genSpecializedAttr = 0;
}

def Ondrix_CfftForward : I32EnumAttrCase<"Forward", 0, "forward">;
def Ondrix_CfftInverse : I32EnumAttrCase<"Inverse", 1, "inverse">;
def Ondrix_CfftDirection
    : I32EnumAttr<"CfftDirection", "complex FFT transform direction",
                  [Ondrix_CfftForward, Ondrix_CfftInverse]> {
  let cppNamespace = "::ondrix::ir";
  let genSpecializedAttr = 0;
}

#endif
