#ifndef ONDRIX_DIALECT_ONDSP_IR_ONDSPOPS_TD
#define ONDRIX_DIALECT_ONDSP_IR_ONDSPOPS_TD

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "ondrix/Dialect/ondsp/IR/OndspAttrs.td"
include "ondrix/Dialect/ondsp/IR/OndspTypes.td"

def Ondsp_AssumeNumericOp
    : Ondsp_Op<"assume_numeric", [Pure, AllTypesMatch<["input", "result"]>]> {
  let summary = "Attach an explicit numeric interpretation to a storage value";
  let arguments = (ins AnyType:$input, Ondsp_NumericAttr:$numeric);
  let results = (outs AnyType:$result);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def Ondsp_ConvertOp : Ondsp_Op<"convert", [Pure]> {
  let summary = "Convert a value between explicit ondsp numeric policies";
  let arguments = (ins AnyType:$input, Ondsp_NumericAttr:$src, Ondsp_NumericAttr:$dst);
  let results = (outs AnyType:$result);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def Ondsp_RoundShiftOp : Ondsp_Op<"round_shift", [Pure]> {
  let summary = "Apply an explicit rounding and shift policy";
  let arguments = (ins AnyType:$input, Ondsp_ScaleAttr:$scale);
  let results = (outs AnyType:$result);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def Ondsp_SatCastOp : Ondsp_Op<"sat_cast", [Pure]> {
  let summary = "Saturating cast under an explicit numeric policy";
  let arguments = (ins AnyType:$input, Ondsp_NumericAttr:$numeric);
  let results = (outs AnyType:$result);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

class Ondsp_BinaryShiftOp<string mnemonic, string opSummary>
    : Ondsp_Op<mnemonic, [Pure]> {
  let summary = opSummary;
  let arguments = (ins AnyType:$lhs, AnyType:$rhs, Ondsp_ScaleAttr:$scale);
  let results = (outs AnyType:$result);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def Ondsp_SatAddShiftOp
    : Ondsp_BinaryShiftOp<"sat_add_shift", "Saturating add plus scaling">;
def Ondsp_SatSubShiftOp
    : Ondsp_BinaryShiftOp<"sat_sub_shift", "Saturating subtract plus scaling">;

def Ondsp_AccZeroOp : Ondsp_Op<"acc_zero", [Pure]> {
  let summary = "Create an all-zero accumulator state";
  let results = (outs Ondsp_Acc:$acc);
  let assemblyFormat = "attr-dict `:` type($acc)";
}

def Ondsp_AccImportOp : Ondsp_Op<"acc_import", [Pure]> {
  let summary = "Import a fixed-point value into an accumulator exactly";
  let description = [{
    Interprets the input using the source fixed-point policy and preserves its
    numeric value in the accumulator domain. The verifier rejects imports that
    would discard fractional bits or cannot represent every source value.
  }];
  let arguments = (ins AnyInteger:$input, Ondsp_FixedAttr:$src);
  let results = (outs Ondsp_Acc:$acc);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

class Ondsp_MacLikeOp<string mnemonic, string opSummary>
    : Ondsp_Op<mnemonic, [Pure, AllTypesMatch<["acc", "result"]>]> {
  let summary = opSummary;
  let arguments = (ins Ondsp_Acc:$acc, AnyInteger:$lhs, AnyInteger:$rhs,
                   Ondsp_FixedAttr:$numeric, Ondsp_ProductAttr:$product);
  let results = (outs Ondsp_Acc:$result);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def Ondsp_MacOp : Ondsp_MacLikeOp<"mac", "Multiply-accumulate"> {
  let description = [{
    Computes the exact fixed-point product selected by `product`, adds it to
    the mathematical accumulator value, and applies `acc.update_overflow` when
    the result is written back to the accumulator storage width. Rounding and
    destination overflow are not part of this operation.
  }];
}

def Ondsp_MacSubOp : Ondsp_MacLikeOp<"mac_sub", "Multiply-subtract accumulate"> {
  let description = [{
    Computes the exact fixed-point product selected by `product`, subtracts it
    from the mathematical accumulator value, and applies
    `acc.update_overflow` when the result is written back to the accumulator
    storage width. Rounding and destination overflow are not part of this
    operation.
  }];
}

def Ondsp_AccAddTermOp
    : Ondsp_Op<"acc_add_term", [Pure, AllTypesMatch<["acc", "result"]>]> {
  let summary = "Add an explicit fixed-point term to an accumulator";
  let description = [{
    Adds the already-computed fixed-point `term` to the mathematical
    accumulator value, then applies `acc.update_overflow` when writing the
    result back to the accumulator storage width. `term_numeric` describes the
    term's signedness, storage width, and fractional position. A term may be a
    single exact product or a product partial sum whose equivalence was proven
    by a preceding transform. This operation is a target-independent
    normalization point for consumers that compute terms separately from
    accumulator updates.
  }];
  let arguments = (ins Ondsp_Acc:$acc, AnyInteger:$term,
                   Ondsp_FixedAttr:$term_numeric);
  let results = (outs Ondsp_Acc:$result);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def Ondsp_AccExportOp : Ondsp_Op<"acc_export", [Pure]> {
  let summary = "Export an accumulator under an explicit destination policy";
  let description = [{
    Converts the accumulator to the destination fixed-point policy. The raw
    accumulator is divided by `2^(acc.frac - dst.frac)` using the selected
    rounding mode, destination-range overflow handling is applied next, and
    the resulting integer is then materialized in the destination storage
    type.
  }];
  let arguments = (ins Ondsp_Acc:$acc, Ondsp_FixedAttr:$dst,
                   Ondsp_RoundingModeAttr:$rounding,
                   Ondsp_OverflowModeAttr:$overflow);
  let results = (outs AnyInteger:$result);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def Ondsp_ReduceMacOp
    : Ondsp_Op<"reduce_mac", [AllTypesMatch<["initial", "result"]>,
                              DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
                              DeclareOpInterfaceMethods<ConditionallySpeculatable>]> {
  let summary = "Ordered rank-1 dot/reduce MAC semantic operation";
  let description = [{
    Accepts two equal-length rank-1 shaped operands and folds their pairwise
    products into `initial` in increasing index order. Fixed-point reductions
    use an `!ondsp.acc` initial/result type, so accumulator width, fractional
    position, signedness, and per-update overflow remain explicit. Floating-
    point reductions use the numeric format as their initial/result type.
    Dynamic lengths require a runtime proof in the selected lowering pipeline.

    This ordered left-fold contract is the portable correctness baseline.
    Consumers may reassociate or vectorize it only when the numeric policy and
    range analysis prove that the transformed order is equivalent.
  }];
  let arguments = (ins AnyType:$initial, AnyType:$lhs, AnyType:$rhs,
                   Ondsp_NumericAttr:$numeric,
                   OptionalAttr<Ondsp_ProductAttr>:$product);
  let results = (outs AnyType:$result);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def Ondsp_CxMulOp : Ondsp_Op<"cx_mul", [Pure]> {
  let summary = "Experimental complex multiply semantic operation";
  let arguments = (ins AnyType:$lhs, AnyType:$rhs, Ondsp_CxLayoutAttr:$layout,
                   Ondsp_NumericAttr:$numeric);
  let results = (outs AnyType:$result);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def Ondsp_CxButterflyOp : Ondsp_Op<"cx_butterfly", [Pure]> {
  let summary = "Experimental complex radix-2 butterfly semantic operation";
  let arguments = (ins AnyType:$a, AnyType:$b, AnyType:$twiddle,
                   Ondsp_CxLayoutAttr:$layout, Ondsp_NumericAttr:$numeric,
                   OptionalAttr<Ondsp_ProductAttr>:$product,
                   OptionalAttr<Ondsp_ScaleAttr>:$scale,
                   DefaultValuedAttr<BoolAttr, "false">:$trivial_twiddle);
  let results = (outs AnyType:$out0, AnyType:$out1);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def Ondsp_FftStageOp : Ondsp_Op<"fft_stage", [Pure]> {
  let summary = "Experimental FFT stage intent operation";
  let arguments = (ins Variadic<AnyType>:$inputs, I64Attr:$stage,
                   Ondsp_CxLayoutAttr:$layout, Ondsp_NumericAttr:$numeric);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def Ondsp_BitrevIndexOp : Ondsp_Op<"bitrev_index", [Pure]> {
  let arguments = (ins Index:$input, I64Attr:$width);
  let results = (outs Index:$result);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def Ondsp_BitrevAddOp : Ondsp_Op<"bitrev_add", [Pure]> {
  let arguments = (ins Index:$base, Index:$step, I64Attr:$width);
  let results = (outs Index:$result);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

#endif
