#ifndef ONDRIX_DIALECT_ONDSP_IR_ONDSPENUMS_TD
#define ONDRIX_DIALECT_ONDSP_IR_ONDSPENUMS_TD

include "mlir/IR/EnumAttr.td"

def Ondsp_Signed : I32EnumAttrCase<"Signed", 0, "signed">;
def Ondsp_Unsigned : I32EnumAttrCase<"Unsigned", 1, "unsigned">;
def Ondsp_Signedness : I32EnumAttr<"Signedness", "fixed-point signedness",
    [Ondsp_Signed, Ondsp_Unsigned]> {
  let cppNamespace = "::ondrix::ondsp";
}

def Ondsp_RoundTrunc : I32EnumAttrCase<"Trunc", 0, "trunc">;
def Ondsp_RoundNearest : I32EnumAttrCase<"Nearest", 1, "nearest">;
def Ondsp_RoundNearestEven : I32EnumAttrCase<"NearestEven", 2, "nearest_even">;
def Ondsp_RoundTowardZero : I32EnumAttrCase<"TowardZero", 3, "toward_zero">;
def Ondsp_RoundingMode : I32EnumAttr<"RoundingMode", "rounding mode",
    [Ondsp_RoundTrunc, Ondsp_RoundNearest, Ondsp_RoundNearestEven,
     Ondsp_RoundTowardZero]> {
  let cppNamespace = "::ondrix::ondsp";
}

def Ondsp_OverflowWrap : I32EnumAttrCase<"Wrap", 0, "wrap">;
def Ondsp_OverflowSaturate : I32EnumAttrCase<"Saturate", 1, "saturate">;
def Ondsp_OverflowUndefined : I32EnumAttrCase<"Undefined", 2, "undefined">;
def Ondsp_OverflowMode : I32EnumAttr<"OverflowMode", "overflow mode",
    [Ondsp_OverflowWrap, Ondsp_OverflowSaturate, Ondsp_OverflowUndefined]> {
  let cppNamespace = "::ondrix::ondsp";
}

def Ondsp_ShiftRoundSaturate
    : I32EnumAttrCase<"ShiftRoundSaturate", 0, "shift_round_saturate">;
def Ondsp_ShiftSaturate
    : I32EnumAttrCase<"ShiftSaturate", 1, "shift_saturate">;
def Ondsp_RoundShiftSaturate
    : I32EnumAttrCase<"RoundShiftSaturate", 2, "round_shift_saturate">;
def Ondsp_ScaleOrder : I32EnumAttr<"ScaleOrder", "scale operation order",
    [Ondsp_ShiftRoundSaturate, Ondsp_ShiftSaturate,
     Ondsp_RoundShiftSaturate]> {
  let cppNamespace = "::ondrix::ondsp";
}

def Ondsp_FpContractOff : I32EnumAttrCase<"Off", 0, "off">;
def Ondsp_FpContractFma : I32EnumAttrCase<"Fma", 1, "fma">;
def Ondsp_FpContractFast : I32EnumAttrCase<"Fast", 2, "fast">;
def Ondsp_FpContractMode : I32EnumAttr<"FpContractMode", "FP contract mode",
    [Ondsp_FpContractOff, Ondsp_FpContractFma, Ondsp_FpContractFast]> {
  let cppNamespace = "::ondrix::ondsp";
}

def Ondsp_CxSplit : I32EnumAttrCase<"Split", 0, "split">;
def Ondsp_CxInterleaved : I32EnumAttrCase<"Interleaved", 1, "interleaved">;
def Ondsp_CxPackedImagHiRealLo
    : I32EnumAttrCase<"PackedI16ImagHiRealLo", 2,
                      "packed_i16_imag_hi_real_lo">;
def Ondsp_CxPackedRealHiImagLo
    : I32EnumAttrCase<"PackedI16RealHiImagLo", 3,
                      "packed_i16_real_hi_imag_lo">;
def Ondsp_ComplexLayout : I32EnumAttr<"ComplexLayout", "complex storage layout",
    [Ondsp_CxSplit, Ondsp_CxInterleaved, Ondsp_CxPackedImagHiRealLo,
     Ondsp_CxPackedRealHiImagLo]> {
  let cppNamespace = "::ondrix::ondsp";
}

#endif
