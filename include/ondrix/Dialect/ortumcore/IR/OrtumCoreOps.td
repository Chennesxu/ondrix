#ifndef ONDRIX_DIALECT_ORTUMCORE_IR_ORTUMCOREOPS_TD
#define ONDRIX_DIALECT_ORTUMCORE_IR_ORTUMCOREOPS_TD

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "ondrix/Dialect/ortumcore/IR/OrtumCoreTypes.td"

def OrtumCore_AccInitOp : OrtumCore_Op<"acc_init", [Pure]> {
  let summary = "Initialize a target accumulator semantic value";
  let results = (outs AnyType:$result);
  let assemblyFormat = "attr-dict `:` type($result)";
}

class OrtumCore_MacOp<string mnemonic, string opSummary>
    : OrtumCore_Op<mnemonic, [Pure]> {
  let summary = opSummary;
  let arguments = (ins AnyType:$acc, AnyType:$lhs, AnyType:$rhs);
  let results = (outs AnyType:$result);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def OrtumCore_DualMacOp : OrtumCore_MacOp<"dual_mac", "Dual-lane MAC semantic op">;
def OrtumCore_MacAddOp : OrtumCore_MacOp<"mac_add", "Multiply-add semantic op">;
def OrtumCore_MacSubOp : OrtumCore_MacOp<"mac_sub", "Multiply-subtract semantic op">;
def OrtumCore_QMacAddOp : OrtumCore_MacOp<"qmac_add", "Q-format multiply-add semantic op">;
def OrtumCore_QMacSubOp : OrtumCore_MacOp<"qmac_sub", "Q-format multiply-subtract semantic op">;

def OrtumCore_AccExtractOp : OrtumCore_Op<"acc_extract", [Pure]> {
  let arguments = (ins AnyType:$acc);
  let results = (outs AnyType:$result);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def OrtumCore_AccImportOp : OrtumCore_Op<"acc_import", [Pure]> {
  let arguments = (ins AnyType:$input);
  let results = (outs AnyType:$result);
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
  let results = (outs AnyType:$state);
  let assemblyFormat = "attr-dict `:` type($state)";
}

def OrtumCore_VecSetModeOp : OrtumCore_Op<"vec_set_mode", [Pure]> {
  let arguments = (ins AnyType:$state);
  let results = (outs AnyType:$result);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

class OrtumCore_VcuUnaryOp<string mnemonic, string opSummary>
    : OrtumCore_Op<mnemonic, [Pure]> {
  let summary = opSummary;
  let arguments = (ins AnyType:$state, AnyType:$input);
  let results = (outs AnyType:$nextState, AnyType:$result);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

class OrtumCore_VcuBinaryOp<string mnemonic, string opSummary>
    : OrtumCore_Op<mnemonic, [Pure]> {
  let summary = opSummary;
  let arguments = (ins AnyType:$state, AnyType:$lhs, AnyType:$rhs);
  let results = (outs AnyType:$nextState, AnyType:$result);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

class OrtumCore_VcfftOp<string mnemonic, string opSummary>
    : OrtumCore_Op<mnemonic, [Pure]> {
  let summary = opSummary;
  let arguments = (ins AnyType:$state, Variadic<AnyType>:$inputs, I64Attr:$variant);
  let results = (outs AnyType:$nextState, Variadic<AnyType>:$outputs);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def OrtumCore_CxMulOp : OrtumCore_VcuBinaryOp<"cx_mul", "Packed complex multiply semantic op">;
def OrtumCore_CxAddOp : OrtumCore_VcuBinaryOp<"cx_add", "Packed complex add semantic op">;
def OrtumCore_CxSubOp : OrtumCore_VcuBinaryOp<"cx_sub", "Packed complex subtract semantic op">;
def OrtumCore_CxDualAddOp : OrtumCore_VcuBinaryOp<"cx_dual_add", "Packed dual add semantic op">;
def OrtumCore_CxDualSubOp : OrtumCore_VcuBinaryOp<"cx_dual_sub", "Packed dual subtract semantic op">;
def OrtumCore_FftTrivialStageOp
    : OrtumCore_VcfftOp<"fft_trivial_stage", "Trivial-twiddle FFT stage semantic op"> {
  let hasVerifier = 1;
}

def OrtumCore_BitrevAddOp : OrtumCore_BinaryOp<"bitrev_add", "Bit-reversed add addressing semantic op">;
def OrtumCore_BitrevSubOp : OrtumCore_BinaryOp<"bitrev_sub", "Bit-reversed subtract addressing semantic op">;

#endif
