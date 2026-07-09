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

#endif
