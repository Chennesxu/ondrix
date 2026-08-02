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
//   nearest_even           ties go to the even quotient (no fixed one-sided
//                          tie direction; aggregate bias depends on the
//                          input distribution and any saturation);
//   nearest_ties_positive  ties go toward +infinity, the add-half-then-shift
//                          convention of common DSP hardware rounding
//                          variants. One instance of this rule — signed i32
//                          input, shift 16, saturating signed i16 result —
//                          has been validated bit-exactly against the
//                          official ITU-T G.191 (05/2024) STL round routine
//                          (openitu/STL tag v2024, src/basop/basop32.c,
//                          SHA-256 6d1babb406677ae6c485cf3f33aa11e85c2007
//                          735a3b49fca3fed12342b2d50b) over the complete
//                          2^32 input domain; no broader basic-operator
//                          compatibility is asserted, and every other
//                          (width, shift, saturation) instance remains a
//                          separate validation obligation;
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

// The declared floating-point evaluation policies. Each mode fixes which
// dynamic FP event graphs a lowering may produce for one call site, and
// therefore which correctness relation the evidence must establish:
//
//   off   every multiply and every add is a separate IEEE-754 operation in
//         the contract's stated order (emitted with NO fast-math flags, so
//         not even mul-add contraction is permitted). Exact contract: any
//         authorized schedule must reproduce the result bit for bit.
//   fma   each multiply-accumulate step is one explicit fused multiply-add
//         EVENT — an infinitely precise product-and-sum with a single
//         rounding (`math.fma` -> `llvm.fma`), never a mere contraction
//         permission on separate operations. Exact contract as well: the
//         event tree is different from off's, but it is pinned, and
//         evidence is again bit-for-bit.
//   fast  a named RELAXATION profile that widens the correctness relation
//         instead of pinning an event tree. It declares exactly two
//         permissions — reassociation of the accumulation graph and
//         mul-add contraction (fastmath<reassoc,contract>) — so schedules
//         may regroup the chain and results are compared under an explicit
//         error characterization, never bit-pinned. It deliberately does
//         NOT assert the absence of NaN/Inf inputs, does not abandon
//         signed-zero observability, and does not license reciprocal
//         division or approximate math functions; each of those would be a
//         separate declaration. The declaration replaces the
//         bit-equivalence obligation only: memory disjointness, bounds,
//         and every other structural legality obligation remain to be
//         discharged by analysis.
//
// Environment contract shared by all three modes: round-to-nearest-even,
// FP exception state unobservable, subnormals preserved (no FTZ/DAZ), and
// NaN/Inf/signed zeros propagate per IEEE 754 unless a mode above states a
// declared exception.
def Ondsp_FpContractOff : I32EnumAttrCase<"Off", 0, "off">;
def Ondsp_FpContractFma : I32EnumAttrCase<"Fma", 1, "fma">;
def Ondsp_FpContractFast : I32EnumAttrCase<"Fast", 2, "fast">;
def Ondsp_FpContractMode : I32EnumAttr<
    "FpContractMode",
    "FP evaluation policy: off preserves separate ordered operations, fma "
    "pins explicit fused multiply-add events, and fast declares the "
    "reassoc+contract relaxation profile",
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
// The Q31 counterpart of packed_i16_imag_hi_real_lo: two signed i32
// components in one i64 container, imaginary in bits 63..32 and real in bits
// 31..0. Only the imag-high spelling exists because only it has an executable
// consumer; a real-high Q31 variant would be unproven surface.
def Ondsp_CxPackedI32ImagHiRealLo
    : I32EnumAttrCase<"PackedI32ImagHiRealLo", 4,
                      "packed_i32_imag_hi_real_lo">;
def Ondsp_ComplexLayout : I32EnumAttr<"ComplexLayout", "complex storage layout",
    [Ondsp_CxSplit, Ondsp_CxInterleaved, Ondsp_CxPackedImagHiRealLo,
     Ondsp_CxPackedRealHiImagLo, Ondsp_CxPackedI32ImagHiRealLo]> {
  let cppNamespace = "::ondrix::ondsp";
}

#endif
