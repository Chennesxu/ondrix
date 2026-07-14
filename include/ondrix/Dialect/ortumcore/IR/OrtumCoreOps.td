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

class OrtumCore_MacOp<string mnemonic, string opSummary>
    : OrtumCore_Op<mnemonic, [Pure, AllTypesMatch<["acc", "result"]>]> {
  let summary = opSummary;
  let arguments = (ins OrtumCore_AccumType:$acc, AnyInteger:$lhs, AnyInteger:$rhs);
  let results = (outs OrtumCore_AccumType:$result);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def OrtumCore_DualMacOp : OrtumCore_MacOp<"dual_mac", "Dual-lane MAC semantic op">;
def OrtumCore_MacAddOp : OrtumCore_MacOp<"mac_add", "Multiply-add semantic op">;
def OrtumCore_MacSubOp : OrtumCore_MacOp<"mac_sub", "Multiply-subtract semantic op">;
def OrtumCore_QMacAddOp : OrtumCore_MacOp<"qmac_add", "Q-format multiply-add semantic op">;
def OrtumCore_QMacSubOp : OrtumCore_MacOp<"qmac_sub", "Q-format multiply-subtract semantic op">;

def OrtumCore_AccExtractOp : OrtumCore_Op<"acc_extract", [Pure]> {
  let arguments = (ins OrtumCore_AccumType:$acc);
  let results = (outs AnyInteger:$result);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def OrtumCore_AccImportOp : OrtumCore_Op<"acc_import", [Pure]> {
  let arguments = (ins AnyInteger:$input);
  let results = (outs OrtumCore_AccumType:$result);
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
  let results = (outs OrtumCore_VecStateType:$state);
  let assemblyFormat = "attr-dict `:` type($state)";
}

def OrtumCore_VecSetModeOp : OrtumCore_Op<"vec_set_mode", [Pure]> {
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
  let arguments = (ins AnyType:$state, AnyType:$input);
  let results = (outs AnyType:$nextState, AnyType:$result);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

class OrtumCore_VecBinaryOp<string mnemonic, string opSummary>
    : OrtumCore_Op<mnemonic, [Pure]> {
  let summary = opSummary;
  let arguments = (ins AnyType:$state, AnyType:$lhs, AnyType:$rhs);
  let results = (outs AnyType:$nextState, AnyType:$result);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

class OrtumCore_PackedFftOp<string mnemonic, string opSummary>
    : OrtumCore_Op<mnemonic, [Pure]> {
  let summary = opSummary;
  let arguments = (ins AnyType:$state, Variadic<AnyType>:$inputs,
                   OrtumCore_FftStageKindAttr:$stage_kind);
  let results = (outs AnyType:$nextState, Variadic<AnyType>:$outputs);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def OrtumCore_CxMulOp : OrtumCore_VecBinaryOp<"cx_mul", "Packed complex multiply semantic op">;
def OrtumCore_CxAddOp : OrtumCore_VecBinaryOp<"cx_add", "Packed complex add semantic op">;
def OrtumCore_CxSubOp : OrtumCore_VecBinaryOp<"cx_sub", "Packed complex subtract semantic op">;
def OrtumCore_CxDualAddOp : OrtumCore_VecBinaryOp<"cx_dual_add", "Packed dual add semantic op">;
def OrtumCore_CxDualSubOp : OrtumCore_VecBinaryOp<"cx_dual_sub", "Packed dual subtract semantic op">;
def OrtumCore_FftTrivialStageOp
    : OrtumCore_PackedFftOp<"fft_trivial_stage", "Trivial-twiddle FFT stage semantic op">;

def OrtumCore_BitrevAddOp : OrtumCore_BinaryOp<"bitrev_add", "Bit-reversed add addressing semantic op">;
def OrtumCore_BitrevSubOp : OrtumCore_BinaryOp<"bitrev_sub", "Bit-reversed subtract addressing semantic op">;

#endif
