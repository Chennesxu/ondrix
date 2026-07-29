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

// Declared taxonomy for signed right shifts. The two directed modes and the
// two nearest modes disagree on ordinary fractional inputs as well; it is
// only the two NEAREST modes that agree everywhere except on exact halves:
//
//   nearest_even           ties go to the even quotient (zero-biased);
//   nearest_ties_positive  ties go toward +infinity, the add-half-then-shift
//                          convention of common DSP hardware rounding
//                          variants. Bit compatibility with a specific basic
//                          operator library profile is NOT asserted here and
//                          remains to be validated against that library's
//                          official routines;
//   toward_negative        floor, no tie case at all;
//   toward_zero            truncation, no tie case at all.
//
// nearest_even and nearest_ties_positive differ exactly on the ties whose
// floor quotient is EVEN, and that happens at both signs: +0.5 has floor
// quotient 0 and gives 0 under nearest_even but 1 under ties-positive, while
// -1.5 has floor quotient -2 and gives -2 versus -1. Evidence for this mode
// must therefore cover even-quotient ties of both signs.
//
// What separates nearest_ties_positive from ties-AWAY-from-zero is a
// different distinction: those two agree on every positive tie and differ on
// every negative one.
def Ondsp_RoundTowardNegative
    : I32EnumAttrCase<"TowardNegative", 0, "toward_negative">;
def Ondsp_RoundNearestTiesPositive
    : I32EnumAttrCase<"NearestTiesPositive", 1, "nearest_ties_positive">;
def Ondsp_RoundNearestEven : I32EnumAttrCase<"NearestEven", 2, "nearest_even">;
def Ondsp_RoundTowardZero : I32EnumAttrCase<"TowardZero", 3, "toward_zero">;
def Ondsp_RoundingMode : I32EnumAttr<
    "RoundingMode",
    "explicit signed right-shift rounding direction or tie rule",
    [Ondsp_RoundTowardNegative, Ondsp_RoundNearestTiesPositive,
     Ondsp_RoundNearestEven, Ondsp_RoundTowardZero]> {
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
