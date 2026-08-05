#ifndef ONDRIX_DIALECT_ORTUMCORE_IR_ORTUMCOREOPS_TD
#define ONDRIX_DIALECT_ORTUMCORE_IR_ORTUMCOREOPS_TD

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "ondrix/Dialect/ortumcore/IR/OrtumCoreAttrs.td"
include "ondrix/Dialect/ortumcore/IR/OrtumCoreTypes.td"

def OrtumCore_AccInitOp : OrtumCore_Op<"acc_init", [Pure]> {
  let summary = "Create a zero-initialized target accumulator semantic value";
  let description = [{
    Produces the numeric zero of the abstract target accumulator domain. A
    target adapter may materialize that value using any equivalent target
    sequence.
  }];
  let results = (outs OrtumCore_AccumType:$result);
  let assemblyFormat = "attr-dict `:` type($result)";
}

class OrtumCore_Q15MacOp<string mnemonic, string opSummary>
    : OrtumCore_Op<mnemonic, [Pure, AllTypesMatch<["acc", "result"]>]> {
  let summary = opSummary;
  let arguments = (ins OrtumCore_AccumType:$acc, I16:$lhs, I16:$rhs);
  let results = (outs OrtumCore_AccumType:$result);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def OrtumCore_MacAddOp
    : OrtumCore_Q15MacOp<"mac_add", "Saturating signed Q15 multiply-add"> {
  let description = [{
    Interprets `lhs` and `rhs` as signed i16 raw values with 15 fractional
    bits, computes their exact signed 16-by-16 full product, and adds that
    product to the signed accumulator. The mathematical update is then
    saturated to the signed 40-bit accumulator range. Export or
    requantization behavior is not part of this operation.
  }];
}
def OrtumCore_MacSubOp
    : OrtumCore_Q15MacOp<"mac_sub", "Saturating signed Q15 multiply-subtract"> {
  let description = [{
    Interprets `lhs` and `rhs` as signed i16 raw values with 15 fractional
    bits, computes their exact signed 16-by-16 full product, and subtracts that
    product from the signed accumulator. The mathematical update is then
    saturated to the signed 40-bit accumulator range. Export or
    requantization behavior is not part of this operation.
  }];
}
def OrtumCore_AccOutOp : OrtumCore_Op<"acc_out", [Pure]> {
  let summary = "Shifted saturating accumulator readout";
  let description = [{
    Reads the abstract signed 40-bit accumulator through the target's
    architectural output path: the value is arithmetically shifted right by
    the static `shift` amount and then saturated to the signed 32-bit range.

    The realized rounding is exactly `toward_negative`, and the saturation
    boundary is the 32-bit result storage, never a narrower destination
    format: a narrower destination composes this operation with an explicit
    saturating cast, which is exact because clamping to a wider range first
    cannot change a subsequent narrower clamp. `shift` models the target's
    output shift register and must lie in [0, 15].
  }];
  let arguments = (ins OrtumCore_AccumType:$acc, I64Attr:$shift);
  let results = (outs I32:$result);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

class OrtumCore_BinaryOp<string mnemonic, string opSummary>
    : OrtumCore_Op<mnemonic, [Pure]> {
  let summary = opSummary;
  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
  let results = (outs AnyType:$result);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def OrtumCore_SatAddOp : OrtumCore_BinaryOp<"sat_add", "Saturating add semantic op">;
def OrtumCore_SatSubOp : OrtumCore_BinaryOp<"sat_sub", "Saturating subtract semantic op">;

def OrtumCore_VecStateInitOp : OrtumCore_Op<"vec_state_init", [Pure]> {
  let summary = "Experimental packed-vector mode state initialization";
  let results = (outs OrtumCore_VecStateType:$state);
  let assemblyFormat = "attr-dict `:` type($state)";
}

def OrtumCore_VecSetModeOp : OrtumCore_Op<"vec_set_mode", [Pure]> {
  let summary = "Experimental packed-vector mode state transition";
  let arguments = (ins OrtumCore_VecStateType:$state,
                       BoolAttr:$saturation,
                       BoolAttr:$round_to_nearest,
                       BoolAttr:$packed_complex,
                       I64Attr:$shift_right,
                       I64Attr:$shift_left);
  let results = (outs OrtumCore_VecStateType:$result);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

class OrtumCore_VecUnaryOp<string mnemonic, string opSummary>
    : OrtumCore_Op<mnemonic, [Pure]> {
  let summary = opSummary;
  let arguments = (ins OrtumCore_VecStateType:$state, I32:$input);
  let results = (outs OrtumCore_VecStateType:$nextState, I32:$result);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

class OrtumCore_VecBinaryOp<string mnemonic, string opSummary>
    : OrtumCore_Op<mnemonic, [Pure]> {
  let summary = opSummary;
  let arguments = (ins OrtumCore_VecStateType:$state, I32:$lhs, I32:$rhs);
  let results = (outs OrtumCore_VecStateType:$nextState, I32:$result);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

class OrtumCore_PackedFftOp<string mnemonic, string opSummary>
    : OrtumCore_Op<mnemonic, [Pure]> {
  let summary = opSummary;
  let arguments = (ins OrtumCore_VecStateType:$state, Variadic<I32>:$inputs,
                   OrtumCore_FftStageKindAttr:$stage_kind);
  let results = (outs OrtumCore_VecStateType:$nextState, Variadic<I32>:$outputs);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def OrtumCore_CxMulOp
    : OrtumCore_VecBinaryOp<"cx_mul", "Experimental packed complex multiply semantic op">;
def OrtumCore_CxAddOp
    : OrtumCore_VecBinaryOp<"cx_add", "Experimental packed complex add semantic op">;
def OrtumCore_CxSubOp
    : OrtumCore_VecBinaryOp<"cx_sub", "Experimental packed complex subtract semantic op">;
def OrtumCore_CxDualAddOp
    : OrtumCore_VecBinaryOp<"cx_dual_add", "Experimental packed dual add semantic op">;
def OrtumCore_CxDualSubOp
    : OrtumCore_VecBinaryOp<"cx_dual_sub", "Experimental packed dual subtract semantic op">;
def OrtumCore_FftTrivialStageOp
    : OrtumCore_PackedFftOp<"fft_trivial_stage",
                            "Experimental trivial-twiddle FFT stage semantic op">;

def OrtumCore_BitrevAddOp : OrtumCore_BinaryOp<"bitrev_add", "Bit-reversed add addressing semantic op">;
def OrtumCore_BitrevSubOp : OrtumCore_BinaryOp<"bitrev_sub", "Bit-reversed subtract addressing semantic op">;

#endif
