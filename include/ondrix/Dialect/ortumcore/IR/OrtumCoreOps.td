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
class OrtumCore_Q31MacOp<string mnemonic, string opSummary>
    : OrtumCore_Op<mnemonic, [Pure, AllTypesMatch<["acc", "result"]>]> {
  let summary = opSummary;
  let arguments = (ins OrtumCore_AccumType:$acc, I32:$lhs, I32:$rhs);
  let results = (outs OrtumCore_AccumType:$result);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def OrtumCore_Q31MacAddOp
    : OrtumCore_Q31MacOp<"q31_mac_add", "Saturating signed Q31 raw-high multiply-add"> {
  let description = [{
    Interprets `lhs` and `rhs` as signed i32 raw values with 31 fractional
    bits, selects the arithmetic HIGH 32 bits of their exact signed product -
    a floor at 32, which leaves a term with 30 fractional bits - and adds that
    term to the signed accumulator, saturating the update to the signed 40-bit
    range. The term's fractional position is the accumulator's own, so this
    family shares one accumulator and one readout path with the Q15
    full-product family rather than declaring its own.
  }];
}
def OrtumCore_Q31MacSubOp
    : OrtumCore_Q31MacOp<"q31_mac_sub", "Saturating signed Q31 raw-high multiply-subtract"> {
  let description = [{
    The `q31_mac_add` term subtracted from the accumulator instead of added,
    with the same raw-high selection and the same saturating update.
  }];
}

class OrtumCore_ScaledBinaryOp<string mnemonic, string opSummary, string noun, string symbol>
    : OrtumCore_Op<mnemonic, [Pure, AllTypesMatch<["lhs", "rhs", "result"]>]> {
  let summary = opSummary;
  let description = [{
    Forms the }] # noun # [{ `lhs `}] # symbol # [{` rhs` exactly - one bit
    wider than the operands, so it never wraps - then shifts it arithmetically
    right by the static `shift` and saturates to the signed 32-bit range. The
    realized rounding is exactly `toward_negative`; there is no rounding
    variant, and at `shift = 0` the operation is a pure saturating }]
    # noun # [{. `shift` models the instruction's scaling immediate and must
    lie in [0, 3].
  }];
  let arguments = (ins I32:$lhs, I32:$rhs, I64Attr:$shift);
  let results = (outs I32:$result);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def OrtumCore_SatShiftAddOp
    : OrtumCore_ScaledBinaryOp<"sat_shift_add", "Saturating scaled add", "sum", "+">;
def OrtumCore_SatShiftSubOp
    : OrtumCore_ScaledBinaryOp<"sat_shift_sub", "Saturating scaled subtract", "difference", "-">;

def OrtumCore_DmacOp
    : OrtumCore_Op<"dmac", [Pure, AllTypesMatch<["lane0", "lane1",
                                                 "out0", "out1"]>]> {
  let summary = "Dual-lane saturating signed Q15 multiply-add step";
  let description = [{
    Advances two independent target accumulators by one full signed Q15
    product each: lane 0 by `lhs0 * rhs0` and lane 1 by `lhs1 * rhs1`, each
    update saturated to the signed 40-bit accumulator range exactly as one
    `mac_add` would be. The pairing is the capability: a target may realize
    both updates with a single dual instruction whose operand halves carry
    the two pairs, but the lanes never interact — the operation denotes
    precisely two independent mac_add updates. Export or requantization
    behavior is not part of this operation; each lane exports through its
    own readout.
  }];
  let arguments = (ins OrtumCore_AccumType:$lane0, OrtumCore_AccumType:$lane1,
                   I16:$lhs0, I16:$rhs0, I16:$lhs1, I16:$rhs1);
  let results = (outs OrtumCore_AccumType:$out0, OrtumCore_AccumType:$out1);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
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

def OrtumCore_CxReduceMacOp
    : OrtumCore_Op<"cx_reduce_mac",
                   [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "Packed complex reduction into a pair of raw i32 accumulators";
  let description = [{
    Folds the pairwise complex products of two equal-length rank-1 memrefs of
    packed signed Q15 values (imaginary in bits 31:16, real in bits 15:0) into
    two raw signed i32 frac30 accumulators, in increasing index order.
    `conjugate` conjugates the right operand.

    Every product is exact and the two cross terms of a component combine
    exactly, so the only requantization is the saturating i32 accumulator
    update, once per component per term. The accumulators are raw i32 rather
    than `!ortumcore.acc` because this domain is 32 bits wide, not 40, and its
    readout is a plain move with no shift and no second saturation.

    The reduction stays one operation down to the target adapter on purpose.
    Its accumulator is a fixed register pair on the unit, so an adapter cannot
    interleave another user of that pair between the seed, the steps and the
    readout, and splitting this into separate init/step/readout operations
    would invite exactly that.
  }];
  let arguments = (ins I32:$initialReal, I32:$initialImag,
                   MemRefRankOf<[I32], [1]>:$lhs,
                   MemRefRankOf<[I32], [1]>:$rhs,
                   UnitAttr:$conjugate);
  let results = (outs I32:$resultReal, I32:$resultImag);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def OrtumCore_ViterbiDecodeOp
    : OrtumCore_Op<"viterbi_decode",
                   [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "Viterbi decoding on the 64-state trellis unit";
  let description = [{
    Decodes as `ondsp.viterbi_decode` does at 16-bit metrics and constraint
    length 7, on the target's trellis unit: 64 states, the branch metrics of
    one symbol pair per stage, and butterflies whose four branches carry one
    metric and its negation -- the shape of a rate-1/2 code exactly when both
    generators tap both ends of the register. Metrics wrap in 16 bits and a
    tie takes the odd predecessor, so the realization's certificate, checked
    here as there, makes the decode exact.

    The decode stays one operation down to the target adapter, as
    `cx_reduce_mac` does: the metrics and decisions are the unit's own fixed
    state, which an adapter can serialize only if it sees every stage,
    renormalization and traceback step together.
  }];
  let arguments = (ins MemRefRankOf<[I16], [1]>:$symbols,
                   MemRefRankOf<[I8], [1]>:$bits,
                   DenseI64ArrayAttr:$polynomials,
                   I64Attr:$symbol_bound,
                   I64Attr:$unreachable_metric,
                   I64Attr:$renormalization_period);
  let extraClassDeclaration = [{
    static constexpr int64_t kConstraintLength = 7;
    static constexpr int64_t kMetricBits = 16;
    /// Whether the unit's butterflies decode this code.
    static bool isTrellisCode(int64_t constraintLength, ::llvm::ArrayRef<int64_t> polynomials);
  }];
  let hasVerifier = 1;
  let assemblyFormat = "$symbols `,` $bits attr-dict `:` type($symbols) `,` type($bits)";
}

def OrtumCore_CxMulConjOp : OrtumCore_Op<"cx_mul_conj", [Pure]> {
  let summary = "Packed complex multiply by the conjugated twiddle";
  let description = [{
    Interprets `value` and `twiddle` as packed complex signed Q15 values
    (imaginary in bits 31:16, real in bits 15:0) and computes
    `value * conj(twiddle)` with exact signed cross products:

    ```
    real = vr*wr + vi*wi     imag = vi*wr - vr*wi     (exact signed i33)
    component(x) = narrow(round(x >> shift))
    ```

    The rounding acts on the exact wide value, so its increment cannot
    overflow, and a `shift` of zero is the identity that bypasses rounding
    entirely; `overflow` then selects two's-complement wrap or i16
    saturation per component. `layout` assigns the components to the
    result halves: `imag_hi` is the standard packing, and `real_hi` swaps
    the halves, which composes with the `cross` butterfly variant to
    realize multiplication by -+j without a separate operation. `shift`
    models the target's product shift and must lie in [0, 31].
  }];
  let arguments = (ins I32:$value, I32:$twiddle, I64Attr:$shift,
                   OrtumCore_CxRoundingAttr:$rounding,
                   OrtumCore_CxOverflowAttr:$overflow,
                   OrtumCore_CxLayoutAttr:$layout);
  let results = (outs I32:$result);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def OrtumCore_CxPowerOp : OrtumCore_Op<"cx_power", [Pure]> {
  let summary = "Packed complex squared magnitude with a shifted saturating readout";
  let description = [{
    Interprets `value` as a packed complex signed Q15 pair and computes
    `re*re + im*im` exactly, then reads the sum back through the target's
    output path:

    ```
    sum    = re*re + im*im                 exact, at most 2^31
    result = sat32(round(sum >> shift))
    ```

    The sum is symmetric in the two halves, so this capability alone among the
    packed complex family is independent of which half carries the real part.
    Rounding acts on the exact wide sum, so its increment cannot overflow, and
    a `shift` of zero bypasses it. The saturation boundary is the 32-bit result
    storage and is reachable there at exactly one input, both halves at the
    component minimum; a narrower destination composes this operation with an
    explicit saturating cast, which is exact because clamping to a wider range
    first cannot change a subsequent narrower clamp. `shift` models the
    target's output shift and must lie in [0, 31].
  }];
  let arguments = (ins I32:$value, I64Attr:$shift, OrtumCore_CxRoundingAttr:$rounding);
  let results = (outs I32:$result);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def OrtumCore_CxBflyOp : OrtumCore_Op<"cx_bfly", [Pure]> {
  let summary = "Packed complex butterfly with per-component scaling";
  let description = [{
    Forms the exact per-component sums and differences of two packed
    complex signed Q15 values (`lhs` = a, `rhs` = b, exact signed i17
    intermediates), then narrows every component through the same
    `shift`/`rounding`/`overflow` pipeline as `cx_mul_conj`. `variant`
    selects the sign pattern:

    ```
    plain: out0 = [ai+bi : ar+br]    out1 = [ai-bi : ar-br]
    cross: out0 = [ai-bi : ar+br]    out1 = [ai+bi : ar-br]
    ```

    With `rhs` holding a `real_hi` product P, the cross variant computes
    exactly `a - j*P` and `a + j*P` under the shared narrowing pipeline.
    `shift` models the target's butterfly scaling and must lie in [0, 1].
  }];
  let arguments = (ins I32:$lhs, I32:$rhs, I64Attr:$shift,
                   OrtumCore_CxRoundingAttr:$rounding,
                   OrtumCore_CxOverflowAttr:$overflow,
                   OrtumCore_CxBflyVariantAttr:$variant);
  let results = (outs I32:$out0, I32:$out1);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

class OrtumCore_BitrevAddressOp<string mnemonic, string opSummary, string op>
    : OrtumCore_Op<mnemonic, [Pure]> {
  let summary = opSummary;
  let description = [{
    Computes `rev32(rev32(lhs) }] # op # [{ rev32(rhs))` where `rev32`
    reverses all 32 bits: carries propagate from the most significant bit
    toward the least significant bit, and the arithmetic wraps around in
    the reversed domain. Integer-exact with no rounding or hidden state;
    this is the address progression of a bit-reversed (FFT reorder) walk.
  }];
  let arguments = (ins I32:$lhs, I32:$rhs);
  let results = (outs I32:$result);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def OrtumCore_BitrevAddOp
    : OrtumCore_BitrevAddressOp<"bitrev_add", "Bit-reversed add addressing", "+">;
def OrtumCore_BitrevSubOp
    : OrtumCore_BitrevAddressOp<"bitrev_sub", "Bit-reversed subtract addressing", "-">;

#endif
