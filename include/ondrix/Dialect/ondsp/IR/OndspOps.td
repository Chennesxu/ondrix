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
  let description = [{
    Requantizes one signed integer value under the complete scale policy.
    With signed input `x` interpreted as a mathematical integer, the exact
    sequence is:

    ```
    shifted = x * 2^pre_shift_left                (exact)
    rounded = round(shifted / 2^post_shift_right) per the rounding mode:
      toward_negative:       floor division
      toward_zero:           floor division, +1 when negative with remainder
      nearest_even:          floor division, +1 when remainder > half, or
                             remainder == half and the floor quotient is odd
      nearest_ties_positive: floor division, +1 when remainder >= half
    result  = narrow(rounded) to the saturate_to storage:
      wrap:     two's-complement truncation
      saturate: clamp to the signed range of saturate_to, then truncate
    ```

    When `saturate_to` matches the input storage width the narrowing step is
    the identity and the operation is a pure explicit rounding shift.
  }];
  let arguments = (ins AnyType:$input, Ondsp_ScaleAttr:$scale);
  let results = (outs AnyType:$result);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def Ondsp_SatCastOp : Ondsp_Op<"sat_cast", [Pure]> {
  let summary = "Saturating cast under an explicit numeric policy";
  let description = [{
    Clamps one signed integer value to the signed range of the numeric
    policy's storage type and produces that storage type:

    ```
    result = min(max(x, smin(storage)), smax(storage))   then truncate
    ```

    Widening a value into a strictly wider storage is exact sign extension.
    The fractional position of the numeric policy documents the fixed-point
    interpretation of the surrounding computation; it does not change the
    clamp arithmetic.
  }];
  let arguments = (ins AnyType:$input, Ondsp_NumericAttr:$numeric);
  let results = (outs AnyType:$result);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def Ondsp_SqrtFixedOp : Ondsp_Op<"sqrt_fixed", [Pure]> {
  let summary = "Explicit fixed-point integer square root";
  let description = [{
    Computes the square root of one non-negative signed integer carrying an
    even-fraction fixed-point value. With input `s` interpreted as a
    mathematical integer, the exact sequence is:

    ```
    root    = floor(sqrt(s))                    (exact integer square root)
    rounded = per the rounding mode:
      toward_negative: root
      nearest_even:    root + 1 when s - root^2 > root, else root
    result  = clamp(rounded, 0, smax(result storage)), then truncate
    ```

    Because `(root + 1/2)^2` is never an integer, the nearest rounding has
    no reachable ties; `nearest_even` names the project-wide policy while
    behaving as plain nearest here. Halving the input fraction position is
    what makes the integer root the exact fixed-point root: an input
    carrying fraction `2f` produces a result carrying fraction `f`.

    A negative input is outside the value domain; producers must establish
    non-negativity structurally (for example as a sum of squares). That
    obligation is not decidable for arbitrary SSA input, so every lowering
    clamps the runtime value to zero from below: an out-of-domain input
    deterministically yields 0 (never an undefined value), and an input
    that is provably negative at compile time fails closed in conversion.
    The first executable profile takes `i64` input to `i16` result under
    the two listed rounding modes; other widths fail closed in lowering.
  }];
  let arguments = (ins AnyType:$input, Ondsp_RoundingModeAttr:$rounding);
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

// A value that carries either one lane or one lane per accumulator lane.
// Coefficients deliberately stay scalar: the per-lane splat is part of the
// declared semantics, so the signature says so instead of hiding it in a
// rewrite.
def Ondsp_LaneInteger : AnyTypeOf<[AnyInteger, VectorOf<[AnyInteger]>]>;

def Ondsp_AccZeroOp : Ondsp_Op<"acc_zero", [Pure]> {
  let summary = "Create an all-zero accumulator state";
  let description = [{
    Produces an accumulator whose every lane is numeric zero. This is one of
    the three operations that accept a multi-lane accumulator.
  }];
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
  let arguments = (ins Ondsp_Acc:$acc, Ondsp_LaneInteger:$lhs, AnyInteger:$rhs,
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

    With a `lanes = W` accumulator the value operand is a `vector<W x storage>`
    and the coefficient stays a scalar broadcast across the lanes. Every lane
    performs the identical declared update on its own accumulator; lanes are
    never combined. This is the only multi-lane accumulator update.
  }];
}

def Ondsp_MacSubOp : Ondsp_MacLikeOp<"mac_sub", "Multiply-subtract accumulate"> {
  let description = [{
    Computes the exact fixed-point product selected by `product`, subtracts it
    from the mathematical accumulator value, and applies
    `acc.update_overflow` when the result is written back to the accumulator
    storage width. Rounding and destination overflow are not part of this
    operation. Multi-lane accumulators are rejected.
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

    With a `lanes = W` accumulator the result is a `vector<W x storage>` and
    every lane is exported independently under the same destination policy.
    This is one of the three operations that accept a multi-lane accumulator.
  }];
  let arguments = (ins Ondsp_Acc:$acc, Ondsp_FixedAttr:$dst,
                   Ondsp_RoundingModeAttr:$rounding,
                   Ondsp_OverflowModeAttr:$overflow);
  let results = (outs Ondsp_LaneInteger:$result);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def Ondsp_ReduceMacOp
    : Ondsp_Op<"reduce_mac", [AllTypesMatch<["initial", "result"]>,
                              DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
                              DeclareOpInterfaceMethods<ConditionallySpeculatable>]> {
  let summary = "Ordered rank-1 dot/reduce MAC semantic operation";
  let description = [{
    Accepts two equal-length rank-1 memrefs or fixed-point vectors and folds
    their pairwise products into `initial` in increasing index order. Tensor
    reductions are not accepted until a bufferization consumer exists, and
    floating-point Vector reduction awaits a dedicated consumer. Fixed-point
    reductions use an `!ondsp.acc` initial/result type, so accumulator width,
    fractional position, signedness, and per-update overflow remain explicit.
    Floating-point memref reductions use the numeric format as their
    initial/result type. Dynamic lengths require a runtime proof in the
    selected lowering pipeline.

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
  let summary = "Packed fixed-point elementwise radix-2 butterfly semantic operation";
  let description = [{
    Implements the target-independent packed complex equation documented by
    `ondrix.butterfly`. The `layout` attribute selects one of two executable
    profiles, and every width below follows from it:

    * `packed_i16_imag_hi_real_lo`: signed Q15 components in an i32
      container, imaginary in bits 31:16 and real in bits 15:0;
    * `packed_i32_imag_hi_real_lo`: signed Q31 components in an i64
      container, imaginary in bits 63:32 and real in bits 31:0.

    The complex product `b*w` is evaluated as exact signed cross-products
    wide enough to hold `br*wr - bi*wi` without wrapping, then
    `product_scale` independently requantizes its real and imaginary
    components back to the component storage. Exact widened additions and
    subtractions form `a + b*w` and `a - b*w`; `output_scale` independently
    requantizes those components before packing.

    Both scaling boundaries explicitly apply left shift, signed right-shift
    rounding, then destination wrap or saturation. Each profile is
    deliberately closed to scalar or matching fixed Vector packed container
    values, full products, a nearest-even saturating product shift of
    `storage width - 1`, and a nearest-even saturating output shift of 1.
    Vector lanes are independent butterflies; the operation does not reorder
    FFT stages or move either requantization boundary.
  }];
  let arguments = (ins AnyType:$a, AnyType:$b, AnyType:$twiddle,
                   Ondsp_CxLayoutAttr:$layout, Ondsp_NumericAttr:$numeric,
                   Ondsp_ProductAttr:$product,
                   Ondsp_ScaleAttr:$product_scale,
                   Ondsp_ScaleAttr:$output_scale);
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
