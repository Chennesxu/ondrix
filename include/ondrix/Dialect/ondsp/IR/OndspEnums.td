#ifndef ONDRIX_DIALECT_ONDSP_IR_ONDSPENUMS_TD
#define ONDRIX_DIALECT_ONDSP_IR_ONDSPENUMS_TD

include "mlir/IR/EnumAttr.td"

def Ondsp_Signed : I32EnumAttrCase<"Signed", 0, "signed">;
def Ondsp_Unsigned : I32EnumAttrCase<"Unsigned", 1, "unsigned">;
def Ondsp_Signedness : I32EnumAttr<"Signedness", "fixed-point signedness",
    [Ondsp_Signed, Ondsp_Unsigned]> {
  let cppNamespace = "::ondrix::ondsp";
}

def Ondsp_ProductFull : I32EnumAttrCase<"Full", 0, "full">;
def Ondsp_ProductHighRaw : I32EnumAttrCase<"HighRaw", 1, "high_raw">;
def Ondsp_ProductSelection
    : I32EnumAttr<"ProductSelection",
                  "fixed-point product result selection (full product or signed raw high half)",
                  [Ondsp_ProductFull, Ondsp_ProductHighRaw]> {
  let cppNamespace = "::ondrix::ondsp";
}

def Ondsp_RoundTowardNegative
    : I32EnumAttrCase<"TowardNegative", 0, "toward_negative">;
def Ondsp_RoundNearestEven : I32EnumAttrCase<"NearestEven", 2, "nearest_even">;
def Ondsp_RoundTowardZero : I32EnumAttrCase<"TowardZero", 3, "toward_zero">;
def Ondsp_RoundingMode : I32EnumAttr<
    "RoundingMode",
    "explicit signed right-shift rounding direction or tie rule",
    [Ondsp_RoundTowardNegative, Ondsp_RoundNearestEven,
     Ondsp_RoundTowardZero]> {
  let cppNamespace = "::ondrix::ondsp";
  let genSpecializedAttr = 0;
}

def Ondsp_OverflowWrap : I32EnumAttrCase<"Wrap", 0, "wrap">;
def Ondsp_OverflowSaturate : I32EnumAttrCase<"Saturate", 1, "saturate">;
def Ondsp_OverflowMode : I32EnumAttr<"OverflowMode", "overflow mode",
    [Ondsp_OverflowWrap, Ondsp_OverflowSaturate]> {
  let cppNamespace = "::ondrix::ondsp";
  let genSpecializedAttr = 0;
}

def Ondsp_FpContractOff : I32EnumAttrCase<"Off", 0, "off">;
def Ondsp_FpContractFma : I32EnumAttrCase<"Fma", 1, "fma">;
def Ondsp_FpContractFast : I32EnumAttrCase<"Fast", 2, "fast">;
def Ondsp_FpContractMode : I32EnumAttr<
    "FpContractMode",
    "FP evaluation policy: off preserves separate operations, fma requires a "
    "fused multiply-add, and fast permits full fast-math transformations",
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
