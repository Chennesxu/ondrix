#ifndef ONDRIX_DIALECT_ORTUMCORE_IR_ORTUMCOREENUMS_TD
#define ONDRIX_DIALECT_ORTUMCORE_IR_ORTUMCOREENUMS_TD

include "mlir/IR/EnumAttr.td"

def OrtumCore_FftStageRadix2 : I32EnumAttrCase<"Radix2", 0, "radix2">;
def OrtumCore_FftStageRadix2Combined
    : I32EnumAttrCase<"Radix2Combined", 1, "radix2_combined">;
def OrtumCore_FftStageKind : I32EnumAttr<"FftStageKind",
    "trivial-twiddle FFT stage kind",
    [OrtumCore_FftStageRadix2, OrtumCore_FftStageRadix2Combined]> {
  let cppNamespace = "::ondrix::ortumcore";
  let genSpecializedAttr = 0;
}

// The packed complex capability admits exactly the two rounding modes the
// target realizes; other ondsp rounding modes are not target capabilities
// and fail closed by not existing here.
def OrtumCore_CxRoundTowardNegative
    : I32EnumAttrCase<"TowardNegative", 0, "toward_negative">;
def OrtumCore_CxRoundNearestTiesPositive
    : I32EnumAttrCase<"NearestTiesPositive", 1, "nearest_ties_positive">;
def OrtumCore_CxRounding : I32EnumAttr<"CxRounding",
    "packed complex per-component rounding",
    [OrtumCore_CxRoundTowardNegative, OrtumCore_CxRoundNearestTiesPositive]> {
  let cppNamespace = "::ondrix::ortumcore";
  let genSpecializedAttr = 0;
}

def OrtumCore_CxOverflowWrap : I32EnumAttrCase<"Wrap", 0, "wrap">;
def OrtumCore_CxOverflowSaturate : I32EnumAttrCase<"Saturate", 1, "saturate">;
def OrtumCore_CxOverflow : I32EnumAttr<"CxOverflow",
    "packed complex per-component overflow handling",
    [OrtumCore_CxOverflowWrap, OrtumCore_CxOverflowSaturate]> {
  let cppNamespace = "::ondrix::ortumcore";
  let genSpecializedAttr = 0;
}

def OrtumCore_CxLayoutImagHi : I32EnumAttrCase<"ImagHi", 0, "imag_hi">;
def OrtumCore_CxLayoutRealHi : I32EnumAttrCase<"RealHi", 1, "real_hi">;
def OrtumCore_CxLayout : I32EnumAttr<"CxLayout",
    "packed complex result half assignment",
    [OrtumCore_CxLayoutImagHi, OrtumCore_CxLayoutRealHi]> {
  let cppNamespace = "::ondrix::ortumcore";
  let genSpecializedAttr = 0;
}

def OrtumCore_CxBflyPlain : I32EnumAttrCase<"Plain", 0, "plain">;
def OrtumCore_CxBflyCross : I32EnumAttrCase<"Cross", 1, "cross">;
def OrtumCore_CxBflyVariant : I32EnumAttr<"CxBflyVariant",
    "packed complex butterfly sign pattern",
    [OrtumCore_CxBflyPlain, OrtumCore_CxBflyCross]> {
  let cppNamespace = "::ondrix::ortumcore";
  let genSpecializedAttr = 0;
}

#endif
