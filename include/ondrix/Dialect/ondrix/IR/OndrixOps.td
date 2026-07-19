#ifndef ONDRIX_DIALECT_ONDRIX_IR_ONDRIXOPS_TD
#define ONDRIX_DIALECT_ONDRIX_IR_ONDRIXOPS_TD

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/DestinationStyleOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "ondrix/Dialect/ondrix/IR/OndrixAttrs.td"
include "ondrix/Dialect/ondrix/IR/OndrixDialect.td"
include "ondrix/Dialect/ondsp/IR/OndspAttrs.td"
include "ondrix/Dialect/ondsp/IR/OndspTypes.td"

def Ondrix_FirOp
    : Ondrix_Op<"fir", [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
                        DeclareOpInterfaceMethods<ConditionallySpeculatable>]> {
  let summary = "Compute one FIR sample from a pre-windowed input";
  let description = [{
    `ondrix.fir` computes one output sample by pairing `input[i]` with
    `coeffs[i]` in increasing index order. Both operands are rank-1 windows
    with equal runtime length. Accumulation starts from numeric zero. Window
    construction, signal boundaries, and streaming state are outside this
    operation. A fixed-point FIR produces an explicit `!ondsp.acc` result;
    destination rounding and overflow are modeled by a subsequent
    `ondsp.acc_export`.
  }];
  let arguments = (ins AnyType:$input, AnyType:$coeffs, Ondsp_NumericAttr:$numeric,
                   OptionalAttr<Ondsp_ProductAttr>:$product);
  let results = (outs AnyType:$result);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def Ondrix_FirFilterOp
    : Ondrix_Op<"fir_filter", [
        DestinationStyleOpInterface,
        DeclareOpInterfaceMethods<ConditionallySpeculatable>,
        NoMemoryEffect,
        TypesMatchWith<"result type matches init type", "init", "result",
                       "$_self">]> {
  let summary = "Compute a valid-boundary FIR output tensor";
  let description = [{
    `ondrix.fir_filter` computes every valid FIR output sample. For input
    length `N` and non-empty coefficient length `K`, `N >= K` and the output
    length is `N - K + 1`. Output sample `n` starts from numeric zero and
    pairs `input[n + k]` with `coeffs[k]` for `k = 0..K-1` in increasing
    order. The `init` tensor provides the destination shape and storage; its
    element values are not part of the mathematical result.

    The first contract accepts ranked one-dimensional tensors only and
    requires `boundary = valid`. This tensor-only surface gives value
    semantics and does not impose an unverifiable memref no-alias promise.
    Other boundary modes, streaming state, stride, and dilation require
    separate contracts.

    Dynamic extent relationships are execution preconditions. The generic
    lowering emits dominating assertions as diagnostic guards, but those
    checks are not observable behavior of an otherwise dead pure operation.

    Fixed-point execution explicitly names the accumulator domain and export
    policy. Floating-point execution uses the policy in `numeric` and must not
    carry fixed-point accumulator or export attributes.
  }];
  let arguments = (ins
      AnyRankedTensor:$input,
      AnyRankedTensor:$coeffs,
      AnyRankedTensor:$init,
      Ondrix_FirBoundaryModeAttr:$boundary,
      Ondsp_NumericAttr:$numeric,
      OptionalAttr<Ondsp_ProductAttr>:$product,
      OptionalAttr<TypeAttrOf<Ondsp_Acc>>:$accumulator,
      OptionalAttr<Ondsp_FixedAttr>:$dst,
      OptionalAttr<Ondsp_RoundingModeAttr>:$rounding,
      OptionalAttr<Ondsp_OverflowModeAttr>:$overflow);
  let results = (outs AnyRankedTensor:$result);
  let extraClassDeclaration = [{
    std::pair<int64_t, int64_t> getDpsInitsPositionRange() {
      return {2, 3};
    }
  }];
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def Ondrix_DotOp
    : Ondrix_Op<"dot", [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
                        DeclareOpInterfaceMethods<ConditionallySpeculatable>]> {
  let summary = "High-level dot/reduce kernel intent";
  let description = [{
    Accepts either two scalar operands or two equal-length rank-1 shaped
    operands. Shaped operands are paired in increasing index order. Dynamic
    lengths require a runtime proof in the selected lowering pipeline. A
    fixed-point dot produces an explicit `!ondsp.acc` result; destination
    rounding and overflow are modeled by a subsequent `ondsp.acc_export`.
  }];
  let arguments = (ins AnyType:$lhs, AnyType:$rhs, Ondsp_NumericAttr:$numeric,
                   OptionalAttr<Ondsp_ProductAttr>:$product);
  let results = (outs AnyType:$result);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def Ondrix_ButterflyOp : Ondrix_Op<"butterfly", [Pure]> {
  let summary = "High-level complex radix-2 butterfly intent";
  let arguments = (ins AnyType:$a, AnyType:$b, AnyType:$twiddle,
                   Ondsp_CxLayoutAttr:$layout, Ondsp_NumericAttr:$numeric,
                   OptionalAttr<Ondsp_ProductAttr>:$product,
                   OptionalAttr<Ondsp_ScaleAttr>:$scale,
                   DefaultValuedAttr<BoolAttr, "false">:$trivial_twiddle);
  let results = (outs AnyType:$out0, AnyType:$out1);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def Ondrix_QuantizeOp : Ondrix_Op<"quantize", [Pure]> {
  let summary = "High-level numeric conversion intent";
  let arguments = (ins AnyType:$input, Ondsp_NumericAttr:$src,
                   Ondsp_NumericAttr:$dst);
  let results = (outs AnyType:$result);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

#endif
