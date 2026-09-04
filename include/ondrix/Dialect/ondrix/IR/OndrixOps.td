#ifndef ONDRIX_DIALECT_ONDRIX_IR_ONDRIXOPS_TD
#define ONDRIX_DIALECT_ONDRIX_IR_ONDRIXOPS_TD

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/DestinationStyleOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/TilingInterface.td"
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
    `ondsp.acc_export`. The current executable windows are memrefs or
    fixed-point vectors. Tensor value semantics belong to output-producing
    algorithm ops such as `ondrix.fir_filter`.
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
        TilingInterface,
        NoMemoryEffect,
        TypesMatchWith<"result type matches init type", "init", "result",
                       "$_self">]> {
  let summary = "Compute a rank-one FIR output tensor";
  let description = [{
    `ondrix.fir_filter` computes a rank-1 FIR output. Every sample starts from
    numeric zero and visits coefficients in increasing `k` order. The `init`
    tensor provides the destination shape and storage; its element values are
    not part of the mathematical result.

    For `boundary = valid`, input length `N` and non-empty coefficient length
    `K` require `N >= K`, the output length is `N - K + 1`, and output sample
    `n` pairs `input[n + k]` with `coeffs[k]` for `k = 0..K-1`.

    For `boundary = full`, both `N` and `K` are non-empty and the output length
    is `N + K - 1`. An optional `output_origin` selects a contiguous subrange
    of that complete output; when absent it is zero and `init` must cover the
    complete output. Local output sample `n` has global index
    `g = output_origin + n` and pairs `coeffs[k]` with
    `input[g + k - (K - 1)]` only when that input index is in `[0, N)`;
    an out-of-range tap performs no accumulator update. This is equivalent to
    zero padding for fixed point, while deliberately avoiding evaluation of
    `0 * Inf` or `0 * NaN` for floating point. This convention makes the fully
    overlapping outputs identical to the valid-boundary equation.

    The first contract accepts ranked one-dimensional tensors only and
    supports explicit `valid` and `full` boundary modes. This tensor-only
    surface gives value semantics and does not impose an unverifiable memref
    no-alias promise. Streaming state, stride, dilation, and other padding
    policies require separate contracts.

    The tiling interface exposes only the output dimension as parallel. A
    valid-boundary tile reads its corresponding input halo. A full-boundary
    tile retains the complete input and carries its global output origin. Both
    retain the complete coefficient tensor. Tiling does not split or
    reassociate the ordered tap updates.

    Dynamic extent relationships are execution preconditions. The generic
    lowering emits dominating assertions as diagnostic guards, but those
    checks are not observable behavior of an otherwise dead pure operation.

    Tensor SSA semantics define the aliasing contract: bufferization must
    preserve read-only input and coefficient values when a destination
    aliases their storage, using an out-of-place result or a dominating
    snapshot as required. The bufferized descriptor convention is an
    execution-test mechanism, not a stable source or C ABI.

    Fixed-point execution explicitly names the accumulator domain and export
    policy. Floating-point execution uses the policy in `numeric` and must not
    carry fixed-point accumulator or export attributes.
  }];
  let arguments = (ins
      AnyRankedTensor:$input,
      AnyRankedTensor:$coeffs,
      AnyRankedTensor:$init,
      Optional<Index>:$output_origin,
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

    ::llvm::SmallVector<::mlir::utils::IteratorType> getLoopIteratorTypes();
    ::llvm::SmallVector<::mlir::Range>
    getIterationDomain(::mlir::OpBuilder &builder);
    ::mlir::FailureOr<::mlir::TilingResult>
    getTiledImplementation(::mlir::OpBuilder &builder,
                           ::llvm::ArrayRef<::mlir::OpFoldResult> offsets,
                           ::llvm::ArrayRef<::mlir::OpFoldResult> sizes);
    ::mlir::LogicalResult getResultTilePosition(
        ::mlir::OpBuilder &builder, unsigned resultNumber,
        ::llvm::ArrayRef<::mlir::OpFoldResult> offsets,
        ::llvm::ArrayRef<::mlir::OpFoldResult> sizes,
        ::llvm::SmallVector<::mlir::OpFoldResult> &resultOffsets,
        ::llvm::SmallVector<::mlir::OpFoldResult> &resultSizes);
    ::mlir::FailureOr<::mlir::TilingResult>
    generateResultTileValue(::mlir::OpBuilder &builder, unsigned resultNumber,
                            ::llvm::ArrayRef<::mlir::OpFoldResult> offsets,
                            ::llvm::ArrayRef<::mlir::OpFoldResult> sizes);
  }];
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def Ondrix_FirDecimateOp
    : Ondrix_Op<"fir_decimate", [
        DestinationStyleOpInterface,
        DeclareOpInterfaceMethods<ConditionallySpeculatable>,
        NoMemoryEffect,
        TypesMatchWith<"result type matches init type", "init", "result",
                       "$_self">]> {
  let summary = "Compute valid phase-zero FIR decimation";
  let description = [{
    `ondrix.fir_decimate` computes a valid, phase-zero FIR decimation over
    rank-1 tensor values. For input length `N`, non-empty coefficient length
    `K`, and static `factor = D >= 2`, the result length is:

      `O = floor((N - K) / D) + 1`

    Output `m` starts from numeric zero and visits taps in increasing `k`
    order:

      `output[m] = ordered_sum_k input[m * D + k] * coeffs[k]`

    The `init` tensor provides destination shape and storage; its element
    values are not part of the mathematical result. The first executable
    contract admits either uniform-Q width with an exact full product and an
    explicit export policy: signed Q15 with a signed frac30 accumulator of at
    least 32 bits, or signed Q31 with the i64/frac62 accumulator the Q31
    reduction family declares. At Q31 the tap sum is carried by the
    accumulator's declared per-update overflow mode rather than by a narrowed
    product, because no exact accumulator exists for `K` products of 62 bits. Input phase,
    padding, interpolation, cross-output polyphase reassociation, and target
    selection are outside this operation. Vectorization is not part of the
    mathematical contract; consumers may batch products only while preserving
    the declared ordered updates.

    The f32 profile carries none of the fixed-point accumulator or export
    attributes: output `m` is exactly the reduction its `#ondsp.fp` contract
    declares over increasing `k`, from `+0.0`. Decimation selects a subset of
    output indices and neither reorders nor merges the terms of any one
    output, so the index relation above is contract-independent and only the
    per-output reduction is contract-indexed.

    Dynamic extent relationships are execution preconditions. Generic
    lowering emits dominating assertions before evaluating any window.
  }];
  let arguments = (ins
      AnyRankedTensor:$input,
      AnyRankedTensor:$coeffs,
      AnyRankedTensor:$init,
      I64Attr:$factor,
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

def Ondrix_CicDecimateOp : Ondrix_Op<"cic_decimate", [Pure]> {
  let summary = "Multiplier-free cascaded integrator-comb decimation";
  let description = [{
    `ondrix.cic_decimate` decimates a static rank-1 signed-Q1.(W-1) tensor,
    for W of 16 or 32, of `R * L` samples to `L` samples (`L` in
    `[1, 4096]`) through `S` integrator stages at the input rate, a rate
    change by `R`, and `S` comb stages with differential delay `M` at the
    output rate. `S` is in
    `[1, 8]`, `R` is a power of two in `[2, 4096]`, and `M` is 1 or 2, so
    the transfer gain `(R*M)^S` is the exact power of two `2^G` with
    `G = S * log2(R*M)`, and the normalizing output shift is exact. All
    state is carried in one declared register width:

    ```text
    W       = storage + G                     at most 64
    s[-1,n] = x[n]                            sign extended to W bits
    s[j,n]  = combine_W(s[j,n-1], s[j-1,n])   j = 0 .. S-1, state starts at 0
    v[m]    = s[S-1, m*R + R-1]               the rate change
    c[-1,m] = v[m]
    c[j,m]  = combine_W(c[j-1,m], -c[j-1,m-M])  c[j,m<0] = 0
    y[m]    = sat_i16(round(c[S-1,m] / 2^G))
    ```

    `combine_W` is `overflow`-declared: the exact sum is `wrap`ped to `W`
    bits or `saturate`d to them. The declaration is load bearing, and it is
    the reason this operation exists. Every integrator has an unbounded
    impulse response, so for a sustained input its state leaves any finite
    width; Hogenauer's result is that two's-complement `wrap` at a width
    that holds the *comb output* still yields the exact result, because the
    comb differences recover the true value modulo `2^W` and that value fits.
    Under `saturate` the same program is simply wrong once an integrator
    reaches the rail. This is the one operation in the catalog whose
    correctness selects the overflow mode rather than merely observing it,
    so no default is inferred: the source must say `wrap`.

    The register width is what the wider input costs: `G` is budgeted at
    `64 - storage`, so 48 bits at Q15 and 32 at Q31, and 33 of the 192
    admissible `(S, R, M)` combinations at Q15 fail closed at Q31 (a
    four-stage cascade at `R = 512`, for instance). That is a declared
    narrowing of the contract, not an implementation limit: carrying the
    cascade past i64 would make the operation generic-scalar-only, which the
    target classes this project maps to do not all support.

    Only the final requantization rounds, and `G < W` makes that shift the
    single boundary; saturation there is declared and unreachable for
    in-range inputs, since the exact DC gain is `2^G`. Decimation is
    phase `R-1` — the last input sample of each block — which is the phase
    that makes the cascade causal with no output-side history.
  }];
  let arguments = (ins AnyRankedTensor:$input,
                   I64Attr:$stages,
                   I64Attr:$rate,
                   I64Attr:$delay,
                   Ondsp_NumericAttr:$numeric,
                   Ondsp_OverflowModeAttr:$overflow,
                   Ondsp_RoundingModeAttr:$rounding);
  let results = (outs AnyRankedTensor:$result);
  let extraClassDeclaration = [{
    /// Bits of transfer gain, `S * log2(R*M)`. Valid once the verifier has
    /// accepted the attributes.
    int64_t getGrowthBits();
  }];
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def Ondrix_FirInterpolateOp
    : Ondrix_Op<"fir_interpolate", [
        DestinationStyleOpInterface,
        DeclareOpInterfaceMethods<ConditionallySpeculatable>,
        NoMemoryEffect,
        TypesMatchWith<"result type matches init type", "init", "result",
                       "$_self">]> {
  let summary = "Compute full phase-zero FIR interpolation";
  let description = [{
    `ondrix.fir_interpolate` computes the complete linear convolution of a
    rank-1 input after phase-zero insertion by static `factor = L`. Let:

      `upsampled[j] = input[j / L]` when `j mod L = 0`, otherwise numeric zero

    For non-empty input length `N` and coefficient length `K`, the result
    length is:

      `O = (N - 1) * L + K`

    Output `m` starts from numeric zero and visits coefficients in increasing
    `k` order. A tap updates the accumulator only when `m >= k`,
    `(m - k) mod L = 0`, and `(m - k) / L < N`:

      `output[m] = ordered_sum_k upsampled[m - k] * coeffs[k]`

    Skipping inserted-zero terms is part of this contract and does not
    authorize reassociation of the remaining updates. The `init` tensor
    provides destination shape and storage; its element values are not part
    of the mathematical result.

    Both executable profiles accept exactly `factor = 2`. The fixed profile
    admits either uniform-Q width with exact full products and an explicit
    export policy — signed Q15 with a signed frac30 accumulator of at least 32
    bits, or signed Q31 with the i64/frac62 accumulator, whose declared
    per-update overflow mode carries the tap sum; the f32 profile carries none of
    those attributes and reduces from `+0.0` under its `#ondsp.fp` contract.
    Under f32 the skip is a declared algorithmic event, not generic
    zero-product elimination: a materialized term is `0.0 * coeffs[k]`, never
    a product of an input sample. The two programs are not interchangeable
    even on all-finite data. Under `fma` an accumulator reaches `-0.0` when a
    real product underflows, and materializing a following `+0.0 * c` term
    with positive `c` rounds it back to `+0.0`. A non-finite coefficient
    separates them as well, since `0.0 * inf` and `0.0 * NaN` are both NaN.
    Stateful block forms, configurable phase, polyphase scheduling, Q31,
    Vector, and target selection are outside this operation.

    Dynamic extent relationships and index arithmetic are execution
    preconditions. Generic lowering emits dominating assertions before
    evaluating any sample.
  }];
  let arguments = (ins
      AnyRankedTensor:$input,
      AnyRankedTensor:$coeffs,
      AnyRankedTensor:$init,
      I64Attr:$factor,
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

def Ondrix_Conv1DOp
    : Ondrix_Op<"conv1d", [
        DestinationStyleOpInterface,
        DeclareOpInterfaceMethods<ConditionallySpeculatable>,
        NoMemoryEffect,
        TypesMatchWith<"result type matches init type", "init", "result",
                       "$_self">]> {
  let summary = "Compute valid one-dimensional convolution or correlation";
  let description = [{
    `ondrix.conv1d` preserves one-dimensional convolution/correlation
    algorithm identity while delegating finite-precision arithmetic to Ondsp.
    It accepts rank-1 tensor values with input length `N`, non-empty kernel
    length `K`, `N >= K`, and result length `N - K + 1`. The `init` tensor
    provides only destination shape and storage; its element values are not
    part of the mathematical result.

    Every output starts from numeric zero and visits `k = 0..K-1` in increasing
    order. For `mode = correlation`:

      `y[n] = ordered_sum_k input[n + k] * kernel[k]`

    For `mode = convolution`:

      `y[n] = ordered_sum_k input[n + k] * kernel[K - 1 - k]`

    The convolution result is therefore the valid subrange of the complete
    linear convolution, in increasing output order. Freezing the update order
    is required because fixed-point saturation and floating-point contraction
    can make alternative real-arithmetic formulations observably different.

    The first contract supports only unit-stride, undilated valid output.
    Padding, same/full boundaries, grouped or multidimensional convolution,
    and conjugating complex correlation require separate extensions.

    Fixed-point execution explicitly names the accumulator domain and export
    policy. Floating-point execution uses `numeric` and must not carry
    fixed-point accumulator or export attributes.
  }];
  let arguments = (ins
      AnyRankedTensor:$input,
      AnyRankedTensor:$kernel,
      AnyRankedTensor:$init,
      Ondrix_Conv1DModeAttr:$mode,
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

def Ondrix_FirStreamOp
    : Ondrix_Op<"fir_stream", [
        DeclareOpInterfaceMethods<ConditionallySpeculatable>,
        NoMemoryEffect]> {
  let summary = "Process one causal FIR stream chunk with explicit history";
  let description = [{
    `ondrix.fir_stream` processes one rank-1 input chunk and returns an
    equal-length output chunk plus the history for the next invocation.
    Coefficient length `K` is non-zero. The input history has length `K - 1`
    and stores samples from oldest to newest.

    Let `extended = concat(state, input)`. Output sample `n` is the ordered FIR
    sample over `extended[n : n + K]`, pairing increasing `k` with
    `coeffs[k]` and starting each accumulation from numeric zero. The next
    state is the final `K - 1` samples of `extended`. This definition covers
    empty chunks and chunks shorter than the history without hidden mutable
    state or an aliasing contract.

    Fixed-point execution explicitly names the accumulator domain and export
    policy. Floating-point execution uses the policy in `numeric` and must not
    carry fixed-point accumulator or export attributes. Stream chunking does
    not authorize reassociation of tap updates. The next state retains the
    source sample storage type and never passes through the output
    requantization policy: history is a copy of input samples, not a derived
    quantity.
  }];
  let arguments = (ins
      AnyRankedTensor:$input,
      AnyRankedTensor:$coeffs,
      AnyRankedTensor:$state,
      Ondsp_NumericAttr:$numeric,
      OptionalAttr<Ondsp_ProductAttr>:$product,
      OptionalAttr<TypeAttrOf<Ondsp_Acc>>:$accumulator,
      OptionalAttr<Ondsp_FixedAttr>:$dst,
      OptionalAttr<Ondsp_RoundingModeAttr>:$rounding,
      OptionalAttr<Ondsp_OverflowModeAttr>:$overflow);
  let results = (outs AnyRankedTensor:$output, AnyRankedTensor:$next_state);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def Ondrix_SosFilterTdf2Op
    : Ondrix_Op<"sos_filter_tdf2", [
        DeclareOpInterfaceMethods<ConditionallySpeculatable>,
        NoMemoryEffect,
        TypesMatchWith<"output type matches input type", "input", "output",
                       "$_self">,
        TypesMatchWith<"next-state type matches state type", "state",
                       "next_state", "$_self">]> {
  let summary = "Process an f32 chunk through a cascade of TDF-II sections";
  let description = [{
    `ondrix.sos_filter_tdf2` is an experimental, value-semantics cascade of
    second-order sections in transposed direct form II. The input and output
    are rank-1 tensors. `coeffs` has shape `[sections, 5]` in the explicit
    order `[b0, b1, b2, a1, a2]`, `scales` has shape `[sections]`, and
    `state` has shape `[sections, 2]` in the order `[z1, z2]`. At least one
    section is required.

    For every input sample, sections are visited in increasing index order.
    Given section input `x`, its scale `s`, coefficients, and current state,
    the operation sequence is:

    ```
    u        = mul(x, s)
    y        = mul_add(u, b0, z1)
    feedback = mul(y, a1)
    z1_next  = add(z2, mul_add(u, b1, feedback))
    z2_next  = mul_add(u, b2, mul(y, a2))
    ```

    `y` becomes the input to the next section. The final section output is
    written to the output chunk. Denominator feedback is added exactly as
    written; a source convention using subtraction must provide negated
    `a1` and `a2` coefficients.

    `mul_add` follows the `numeric` contraction policy: `off` emits the stated
    separate multiply and add, and `fma` one fused multiply-add. `fast` is not
    admitted by this executable profile, because no realization gate covers the
    selected biquad event graph; the verifier refuses it rather than emitting a
    permission no site can consume. The standalone `mul` and `add` steps stay
    distinct operations in both admitted modes and carry no fast-math flag in
    either. Empty input returns an empty output and the unchanged state.

    This first contract supports `f32` only. Fixed-point SOS requires an
    independent contract because feedback-state quantization and scaling
    points are part of its observable semantics.
  }];
  let arguments = (ins
      AnyRankedTensor:$input,
      AnyRankedTensor:$coeffs,
      AnyRankedTensor:$scales,
      AnyRankedTensor:$state,
      Ondsp_NumericAttr:$numeric);
  let results = (outs AnyRankedTensor:$output, AnyRankedTensor:$next_state);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def Ondrix_SosFilterDf2FixedOp
    : Ondrix_Op<"sos_filter_df2_fixed", [
        DeclareOpInterfaceMethods<ConditionallySpeculatable>,
        NoMemoryEffect,
        TypesMatchWith<"output type matches input type", "input", "output",
                       "$_self">,
        TypesMatchWith<"next-state type matches state type", "state",
                       "next_state", "$_self">]> {
  let summary = "Process a fixed-point chunk through uniform-Q DF-II sections";
  let description = [{
    `ondrix.sos_filter_df2_fixed` is an experimental, value-semantics cascade
    of signed uniform-Q-format second-order sections in direct form II. The
    input and output are rank-1 tensors. `coeffs` has shape `[sections, 5]` in
    the explicit order `[b0, b1, b2, a1, a2]`, `scales` has shape
    `[sections]`, and `state` has shape `[sections, 2]` in `[d1, d2]` order.
    Coefficients are pre-normalized to `a0 = 1`; normalization is outside the
    operation. At least one section is required.

    For section input `x`, scale `s`, and current state, the exact ordered
    recurrence is:

    ```
    state_acc = mac(mac(mac(0, x, s), d1, a1), d2, a2)
    next_d1   = export(state_acc, state_rounding, state_overflow)
    output_acc = mac(mac(mac(0, next_d1, b0), d1, b1), d2, b2)
    y         = export(output_acc, output_rounding, output_overflow)
    next_d2   = d1
    ```

    Every MAC applies `accumulator.update_overflow`. Both exports return to
    `numeric`; state and output quantization policies are independent.
    Denominator feedback is additive, so subtractive source conventions must
    provide negated `a1/a2`. Samples and sections execute in increasing order;
    `y` feeds the next section. Empty input returns unchanged state.

    The initial executable profiles are exactly signed Q15/full with an
    i40/frac30 accumulator and signed Q31/full with an i64/frac62 accumulator.
    Raw-high products, nonuniform formats, target selection, and compatibility
    with an external kernel ABI are outside this contract.
  }];
  let arguments = (ins
      AnyRankedTensor:$input,
      AnyRankedTensor:$coeffs,
      AnyRankedTensor:$scales,
      AnyRankedTensor:$state,
      Ondsp_FixedAttr:$numeric,
      Ondsp_ProductAttr:$product,
      TypeAttrOf<Ondsp_Acc>:$accumulator,
      Ondsp_RoundingModeAttr:$state_rounding,
      Ondsp_OverflowModeAttr:$state_overflow,
      Ondsp_RoundingModeAttr:$output_rounding,
      Ondsp_OverflowModeAttr:$output_overflow);
  let results = (outs AnyRankedTensor:$output, AnyRankedTensor:$next_state);
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
    Ranked tensors are reserved for output-producing algorithm operations;
    current shaped dot consumers accept memrefs or fixed-point vectors.
  }];
  let arguments = (ins AnyType:$lhs, AnyType:$rhs, Ondsp_NumericAttr:$numeric,
                   OptionalAttr<Ondsp_ProductAttr>:$product);
  let results = (outs AnyType:$result);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def Ondrix_ButterflyOp : Ondrix_Op<"butterfly", [Pure]> {
  let summary = "Packed-Q15 radix-2 butterfly intent";
  let description = [{
    Computes one target-independent radix-2 butterfly over packed signed-Q15
    complex values. Bits 31:16 contain the imaginary component and bits 15:0
    contain the real component.

    For `b = br + j*bi` and `w = wr + j*wi`, the exact complex product is
    formed in signed i33/Q30 intermediates:

      `pr = br*wr - bi*wi`
      `pi = br*wi + bi*wr`

    `product_scale` requantizes each product component to signed Q15. The
    butterfly then forms exact signed i17 components `a + t` and `a - t`;
    `output_scale` requantizes those components before packing the two i32
    results. Scaling applies left shift, signed right-shift rounding, then
    wrap or saturation in that order.

    The current executable profile requires full products, nearest-even
    saturating Q30-to-Q15 product scaling by 15 bits, and nearest-even
    saturating output scaling by one bit. Twiddle classification is derived
    by analysis and is not a user-authored legality promise.
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

def Ondrix_CfftOp : Ondrix_Op<"cfft", [Pure]> {
  let summary = "Static packed fixed-point radix-2 CFFT";
  let description = [{
    Computes one static power-of-two radix-2 decimation-in-time CFFT over
    packed signed fixed-point complex values in natural index order. The
    `layout` attribute selects the profile, which fixes the container and the
    supported extents:

    * `packed_i16_imag_hi_real_lo`: signed Q15, matching `tensor<Nxi32>`
      input and result, `N` in `[4, 1024]`;
    * `packed_i32_imag_hi_real_lo`: signed Q31, matching `tensor<Nxi64>`
      input and result, `N` in `[4, 64]`;
    * `interleaved` with a floating-point `numeric`: one complex value as two
      adjacent elements, real first, so the transform maps `tensor<2Nxf32>`
      to itself, `N` in `[4, 1024]`.

    Each packed element uses the layout and finite-precision butterfly
    contract of `ondsp.cx_butterfly` under the same profile.

    The transform recursively decomposes even and odd input indices, then
    combines the two half transforms in increasing frequency order:

      `(y[k], y[k + N/2]) = butterfly(even[k], odd[k], W_N^k)`

    Forward direction uses `W_N^k = exp(-2*pi*j*k/N)`; inverse direction
    uses `conj(W_N^k)`. Each twiddle component is quantized once from the
    real-valued definition with round-half-even signed Q1.f and declared
    saturation (`+1.0` and `+j` use the positive maximum, `-1.0` and `-j`
    use the exact negative minimum).

    The two profiles obtain those integers differently, and the difference is
    part of the contract. Q15 quantizes binary64 estimates in the compiler
    and is admissible only when the estimate is at least `2^-20` Q15 LSB from
    every rounding tie, so the emitted constants provably equal the
    real-valued definition under the declared binary64/libm error budget
    (more than three orders of magnitude below the guard). At Q31 that guard
    is `2^-51` absolute, only about four times a binary64 evaluation error,
    so it can no longer certify an in-compiler estimate; the Q31 tables are
    instead evaluated offline at 50 decimal digits and frozen as literal
    constants that the conversion reads directly.

    Four points use `[1, -j]` forward; eight points use
    `[1, 23170-j23170, -j, -23170-j23170]` forward at Q15 and
    `[1, 1518500250-j1518500250, -j, -1518500250-j1518500250]` forward at
    Q31, with the conjugate tables in inverse direction.

    Each butterfly applies the declared product and output requantization
    independently. A size-N transform therefore has `log2(N)` observable
    scaling boundaries; they must not be collapsed into one final
    real-arithmetic normalization. With the current one-bit output scale at
    every stage, inverse direction includes the conventional `1/N`
    normalization.

    The executable profiles are deliberately limited to static power-of-two
    extents within each profile's declared range, explicit forward or inverse
    direction, an imag-high/real-low packed layout, and the executable
    butterfly's scale policies. The Q15 profile uses exact full products; the
    Q31 profile is frozen to the raw-high selection, and any other product
    selection is refused at the transform level (the full product stays
    admitted only on the operation-level butterfly, where its exact-carrier
    gate lives); at Q15 that
    includes the packed target inventory (toward_negative or
    nearest_ties_positive rounding, wrap or saturating narrowing, output
    shift 0 or 1), which the unrolled lowering propagates unchanged into
    every stage butterfly. Under the inventory profile the transform
    equation is the target equation: the combines of size 2 and 4 are the
    exact unit stages - the twiddle is exactly one or exactly -j, so there
    is no product stage at all and only the per-stage output scaling
    applies - and every larger combine pairs legs j and j + H/2 on the
    shared requantized product of the single twiddle W(2H, j), the first
    leg through the plain combine and the second through the cross combine
    (a -+ j*t). -j never appears as a multiplier anywhere; this staged
    equation IS the inventory equation, not an optimization of the
    nearest-even one, whose unpaired uniformly-quantized equation is
    unchanged. The opt-in loop-form and Vector-batched CFFT lowerings
    follow the packed profile at both widths; the Q15 inventory profile
    is the exception, always converting as the paired scalar form.
    Dynamic planning, other sizes, and other numeric profiles require
    separate extensions.

    The interleaved floating-point profile declares none of `product`,
    `product_scale`, and `output_scale`, because the program it denotes
    contains no requantization boundary; its one contract axis is the
    format's own `contract` mode, spent on the two multiply-add events of
    each butterfly's complex product. Stage twiddles are the binary32
    roundings of the same real-valued definition, evaluated in the compiler
    without a tie guard for the reason stated on `ondrix.dct`'s f32
    coefficients.

    The stage legs whose twiddle is exactly one (index 0) and exactly `-+j`
    (index `n/4`) CONTAIN NO PRODUCT EVENT. That is part of this contract,
    not a strength reduction a later pass may undo or reintroduce: `0.0 * x`
    is not the identity when `x` is infinite or a negative zero, and the
    declared floating-point environment propagates both per IEEE 754. A leg
    written as `a +- w*b` with `w = 1` would turn a `(+inf, -0.0)` operand
    into a NaN imaginary component; the declared leg returns `-0.0`.

    The inverse direction's `1/N` is a single final multiply rather than
    `log2(N)` stage shifts. This is a DECLARED CHOICE, not a collapse
    justified by equivalence: `2^-log2(N)` is exactly representable, so the
    two orders agree bit for bit only while no intermediate leaves the
    format's range. They diverge on overflow -- a size-4 inverse of four
    `2^126` values reaches `2^128` before the final scaling and returns an
    infinity where per-stage halving would have returned `2^126` -- and on
    underflow to subnormal. The single multiply is what the contract names;
    the fixed-point profiles, whose stage scaling is not optional, cannot
    make even the bounded-range claim.

    The opt-in loop-form lowering covers this profile too, and emits the
    exact unit and quarter-turn phases as their own loops rather than
    branching inside one, so it stays bit-identical to the unrolled form.
    The Vector-batched lowering does not cover it.
  }];
  let arguments = (ins AnyRankedTensor:$input,
                   Ondrix_CfftDirectionAttr:$direction,
                   Ondsp_CxLayoutAttr:$layout,
                   Ondsp_NumericAttr:$numeric,
                   OptionalAttr<Ondsp_ProductAttr>:$product,
                   OptionalAttr<Ondsp_ScaleAttr>:$product_scale,
                   OptionalAttr<Ondsp_ScaleAttr>:$output_scale);
  let results = (outs AnyRankedTensor:$result);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def Ondrix_RfftOp : Ondrix_Op<"rfft", [Pure]> {
  let summary = "Static packed real-to-Hermitian FFT";
  let description = [{
    Computes one static power-of-two real FFT from signed fixed-point real
    input to compact natural-order Hermitian bins in the declared packed
    complex layout. The layout selects the profile exactly as `ondrix.cfft`
    does: packed Q15 maps `tensor<Nxi16>` to `tensor<(N/2+1)xi32>` for `N` in
    `[8, 1024]`; packed Q31 maps `tensor<Nxi32>` to `tensor<(N/2+1)xi64>` for
    `N` in `[8, 64]`, the frozen-table ceiling. The `interleaved` layout with a
    floating-point `numeric` maps `tensor<Nxf32>` to `tensor<(N+2)xf32>` --
    `N/2+1` complex bins of two adjacent elements each -- for `N` in
    `[8, 1024]`, and declares none of the three requantization attributes,
    exactly as `ondrix.cfft` describes.

    The reference schedule embeds every real input `x[n]` as
    `x[n] + j*0`, evaluates the size-N staged radix-2 CFFT equation, and
    retains bins `0..N/2`. It uses the same twiddle tables and the same
    product and output requantization boundaries as `ondrix.cfft` at the
    matching width. For example, the size-16 forward Q15 twiddle pairs
    `(real, imaginary)` are:

    ```
    [(32767, 0), (30274, -12540), (23170, -23170), (12540, -30274),
     (0, -32768), (-12540, -30274), (-23170, -23170), (-30274, -12540)]
    ```

    The one-bit output scale at each of `log2(N)` stages has nominal cumulative
    scale `1/N`. The sequence of saturating stage operations is the exact
    contract; it must not be collapsed into one final division.

    The imaginary components of the DC and Nyquist results are canonically
    zero. Half-size packing and real-pair recombination are optimizations of
    this contract, not part of the reference schedule.

    The executable profiles are limited to static power-of-two extents in
    each profile's declared range, the imag-high/real-low packed layouts,
    and the executable butterfly's scale policies: exact full products at
    Q15, and only the raw-high selection at Q31.
  }];
  let arguments = (ins AnyRankedTensor:$input,
                   Ondsp_CxLayoutAttr:$layout,
                   Ondsp_NumericAttr:$numeric,
                   OptionalAttr<Ondsp_ProductAttr>:$product,
                   OptionalAttr<Ondsp_ScaleAttr>:$product_scale,
                   OptionalAttr<Ondsp_ScaleAttr>:$output_scale);
  let results = (outs AnyRankedTensor:$result);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def Ondrix_IrfftOp : Ondrix_Op<"irfft", [Pure]> {
  let summary = "Static packed Hermitian-to-real inverse FFT";
  let description = [{
    Computes one static power-of-two inverse real FFT from compact
    natural-order Hermitian bins to signed fixed-point real output. The
    layout selects the profile exactly as `ondrix.rfft` does: packed Q15
    maps `tensor<(N/2+1)xi32>` to `tensor<Nxi16>` for `N` in `[8, 1024]`;
    packed Q31 maps `tensor<(N/2+1)xi64>` to `tensor<Nxi32>` for `N` in
    `[8, 64]`; interleaved f32 maps `tensor<(N+2)xf32>` to `tensor<Nxf32>`
    for `N` in `[8, 1024]`, declaring none of the three requantization
    attributes and applying the `1/N` normalization once at the end.

    The reference schedule canonicalizes the imaginary components of the DC
    and Nyquist bins to zero, copies bins `1..N/2-1`, and reconstructs bin
    `N-k` by saturating conjugation of bin `k` at the storage width:
    negating the exact storage minimum saturates to the storage maximum. The
    reconstructed size-N spectrum is evaluated by the staged inverse CFFT
    equation, and the real component of each output is returned. Inverse
    twiddles come from the same tables as `ondrix.cfft`; for example, the
    size-16 inverse Q15 twiddle pairs `(real, imaginary)` are:

    ```
    [(32767, 0), (30274, 12540), (23170, 23170), (12540, 30274),
     (0, 32767), (-12540, 30274), (-23170, 23170), (-30274, 12540)]
    ```

    Every inverse CFFT product and output requantization boundary remains
    observable. The one-bit output scale at each stage has nominal cumulative
    scale `1/N`, but the stage operations must not be collapsed. Consequently,
    `irfft(rfft(x))` is the sequential finite-precision program and is not an
    identity rewrite.

    The executable profiles are limited to static power-of-two extents in
    each profile's declared range, the imag-high/real-low packed layouts,
    and the executable butterfly's scale policies: exact full products at
    Q15, and only the raw-high selection at Q31.
  }];
  let arguments = (ins AnyRankedTensor:$input,
                   Ondsp_CxLayoutAttr:$layout,
                   Ondsp_NumericAttr:$numeric,
                   OptionalAttr<Ondsp_ProductAttr>:$product,
                   OptionalAttr<Ondsp_ScaleAttr>:$product_scale,
                   OptionalAttr<Ondsp_ScaleAttr>:$output_scale);
  let results = (outs AnyRankedTensor:$result);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def Ondrix_RfftRadix4SplitOp : Ondrix_Op<"rfft_radix4_split", [Pure]> {
  let summary = "Half-size radix-4 split real FFT with floor shifts";
  let description = [{
    Computes one static thirty-two-point forward real FFT from signed-Q1.15
    `tensor<32xi16>` input to compact natural-order `tensor<17xi32>` Hermitian
    bins in the declared packed complex layout, using the frozen half-size
    schedule. This is a second, deliberately different real-spectrum contract
    from `ondrix.rfft`; the two are not interchangeable.

    The schedule views the input as sixteen complex values
    `z[m] = x[2m] + j*x[2m+1]` and evaluates:

    1. a length-16 complex FFT of exactly two radix-4 stages. Stage one
       floor-shifts every input right by two, combines with exact widened
       arithmetic, and requantizes each twiddled product by an exact i32
       cross product followed by a floor shift right by sixteen. Stage two
       combines with SATURATING i16 adds and subtracts (the only saturation
       points in the schedule) followed by independent one-bit floor shifts.
       Within each radix-4 group the W^1 and W^2 results swap positions so
       plain binary bit reversal restores natural order;
    2. binary bit reversal of the sixteen complex values;
    3. a real split stage combining `Z[k]` and `Z[16-k]` with frozen Q15
       coefficient tables through exact i32 four-term cross products and a
       floor shift right by sixteen, producing bins `1..15`; DC and Nyquist
       are the one-bit floor-shifted sum and difference of the components of
       `Z[0]`, with imaginary parts exactly zero.

    Every narrowing outside the stage-two saturating combines is proven
    in-range by the recorded numeric-model analysis, so the lowering omits
    those provably inactive clamps without changing any result bit. All
    rounding in the schedule is floor (toward negative); nothing rounds to
    nearest.

    The output numeric interpretation is signed Q6.10: bins approximate the
    unnormalized thirty-two-point DFT of the Q1.15 input. The executable
    profile is limited to this exact schedule: static N=32, the
    imag-high/real-low packed layout, signed Q1.15 input, signed Q6.10
    output, and exact full products. Other extents, layouts, formats, and
    inverse transforms fail closed.
  }];
  let arguments = (ins AnyRankedTensor:$input,
                   Ondsp_CxLayoutAttr:$layout,
                   Ondsp_NumericAttr:$input_numeric,
                   Ondsp_NumericAttr:$output_numeric,
                   Ondsp_ProductAttr:$product);
  let results = (outs AnyRankedTensor:$result);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def Ondrix_RfftSplitOp : Ondrix_Op<"rfft_split", [Pure]> {
  let summary = "Half-size packed-Q31 real-spectrum split stage";
  let description = [{
    Consumes the natural-order output of a half-size packed-Q31 complex FFT
    and produces the compact real spectrum of the real signal that transform
    packed. This is a THIRD real-spectrum contract, deliberately different
    from `ondrix.rfft`, which emits the `N/2+1` Hermitian bins of the
    full-length staged transform, and from `ondrix.rfft_radix4_split`, which
    is the Q15 whole-transform profile. None of the three is interchangeable
    with another. Forward only.

    Input and result are matching `tensor<Nxi64>` values for power-of-two `N`
    in `[4, 32]`, where `N` is the complex point count of the producing
    half-size transform. Both carry the imag-high/real-low packed layout with
    signed `i32` components at `frac = 31` and exact full products.

    Every bin `k` pairs with its mirror `m = N-k` under the Q2.30 twiddle
    pair `(ws, wc) = (sin, cos)(pi*k/N)`. With `xr`/`xi` the components of
    `input[k]`, `mr`/`mi` those of `input[m]`, `wrap32` two's-complement
    32-bit wrapping, `tz` toward-zero and `floor` toward-negative rounding,
    and `low32` truncation to the low thirty-two bits, each `k` in
    `1..N/2-1` is:

    ```
    ar = sat32( floor((xr - mr) / 2) )
    sr = tz( wrap32(xr + mr) / 2 )
    ai = tz( wrap32(xi - mi) / 2 )
    si = tz( wrap32(xi + mi) / 2 )
    p1 = tz( (ws * ar) / 2^30 )
    p2 = tz( (wc * si) / 2^30 )
    p3 = tz( (ws * si) / 2^30 )
    p4 = tz( (wc * ar) / 2^30 )
    Y_re[k]   = low32( tz((sr - p1 + p2) / 2) )
    Y_im[k]   = low32( tz((ai - p3 - p4) / 2) )
    Y_re[N-k] = low32( tz((sr + p1 - p2) / 2) )
    Y_im[N-k] = low32( tz((-ai - p3 - p4) / 2) )
    ```

    Bin `N/2` is its own mirror, so `ar` and `ai` are exactly zero and the
    schedule collapses to two terms:

    ```
    sr = tz( wrap32(xr + xr) / 2 )
    si = tz( wrap32(xi + xi) / 2 )
    Y_re[N/2] = low32( tz((sr + tz((wc * si) / 2^30)) / 2) )
    Y_im[N/2] = low32( tz((-tz((ws * si) / 2^30)) / 2) )
    ```

    DC is `Y_re[0] = tz( wrap32(x_re[0] + x_im[0]) / 2 )` with `Y_im[0] = 0`.
    The Nyquist bin is deliberately dropped: this profile emits exactly `N`
    compact bins, which is part of the contract and not an omission.

    The four products are exact 64-bit multiplies scaled afterwards, never
    a narrowed operand. The `floor` at `ar` is the schedule's single
    toward-negative site and its clamp is provably inert, because a 33-bit
    difference halved always fits `i32`; every other rounding decision is
    toward zero. The `wrap32` inside the halvings is reachable at peak plus
    peak and is contract, not undefined behavior.

    The `low32` truncations carry no saturation because none is reachable.
    The wrapped halvings bound `|sr|`, `|ai|`, and `|si|` by `2^30`, while
    `ar` is an unwrapped 33-bit difference halved and so bounds only by
    `2^31` - the same fact that makes its clamp inert. Hence `|p1|` and
    `|p4|` bound by `2*|w|`, and each output combine bounds by
    `2^30 + (2*|ws| + |wc|)` or by its symmetric `2^30 + (|ws| + 2*|wc|)`
    form. Since `ws` and `wc` lie on the Q2.30 circle, `max(2s + c)` is
    `sqrt(5) * 2^30`, so every combine stays strictly below
    `(1 + sqrt(5)) * 2^30 + 3 < 2^32` and the final toward-zero halving
    always fits `i32`: the truncation is exact for all inputs. The largest
    combine magnitude the extremes reach is `3.2352 * 2^30`.

    Twiddles are the frozen signed Q2.30 words of `sin`/`cos(pi*k/32)`;
    smaller extents stride the same quadrant by `32/N`. Other extents,
    layouts, numeric formats, product selections, and the inverse direction
    fail closed.
  }];
  let arguments = (ins AnyRankedTensor:$input,
                   Ondsp_CxLayoutAttr:$layout,
                   Ondsp_NumericAttr:$input_numeric,
                   Ondsp_NumericAttr:$output_numeric,
                   Ondsp_ProductAttr:$product);
  let results = (outs AnyRankedTensor:$result);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def Ondrix_DctOp : Ondrix_Op<"dct", [Pure]> {
  let summary = "Static signed-Q15 type-II DCT";
  let description = [{
    `ondrix.dct` computes the unnormalized type-II discrete cosine
    transform of a static rank-1 signed-Q1.15 tensor, for power-of-two
    extents `N = 2^m` in `[4, 64]`:

    ```text
    X[k] = round_shift(sum_n x[n] * c[k][n], 16 + m),  k = 0..N-1
    c[k][n] = q15(cos(pi * (2n+1) * k / (2N)))
    ```

    Coefficients are generated at compile time with the project-wide
    tie-guarded round-half-even Q1.15 quantization (declared saturation at
    the `k = 0` cosine of one; a 50-digit sweep of all 5456 coefficients
    for the supported extents shows a worst tie margin of 0.0044 LSB, and
    the guard remains the fail-closed backstop). Products and the sum are
    exact in i64 — `|sum| <= N * 32767 * 32768 <= 2^36 - 2^21` — so the reduction has no
    reachable intermediate overflow and its association is value-neutral;
    the single explicit boundary is the final saturating shift by
    `16 + m`, whose rounding is declared: absent or `nearest_even` (the
    default), `toward_negative`, or `nearest_ties_positive` — each names
    a different function, never a hint. The output reading is the
    unnormalized transform at `frac = 14 - m` (for example Q5.11 at
    `N = 8`). The exported value lies in `[-16384, 16383]` — the negative
    bound is reached exactly once, by the all-minimum input on the `k = 0`
    row through the `-16383.5` half-even tie — so the declared saturation
    is provably unreachable. Normalization conventions are the consumer's
    explicit follow-up quantization, not part of this contract.

    The executable profile admits signed Q1.15 and signed Q1.31 for
    `input_numeric`; `output_numeric` is the same storage with
    `frac = W - 2 - m` derived from the result extent, because one sign bit
    and `m` integer bits are spent on the unnormalized growth at either
    width.

    At Q31 the row sum does not fit i64 and the boundary moves onto the
    product, as it does for `ondrix.matmul`: the exact products of `2^m`
    Q1.31 terms would need `62 + m` bits, so each term is requantized by the
    derived smallest shift `p = max(0, 2*(W-1) + floor(log2 N) - 62)`, which
    is `m` at every admitted Q31 extent and `0` at every Q15 one. Its
    rounding is declared through `product_rounding`, required exactly where
    the boundary exists and refused where it does not. The narrowing absorbs
    the extent dependence, so the export shift is a constant 32 at Q31 while
    it is `16 + m` at Q15, and the exported value lies in `[-2^30, 2^30)`
    just as the Q15 reading lies in `[-2^14, 2^14)`.

    Q31 coefficients are quantized under a WIDER admissibility guard than
    Q15, `2^-13` LSB against `2^-20`, and that is not a relaxation. The
    binary64 representation granularity measured in LSB units grows with the
    format, so the Q15 constant would stand a single bit above it at Q31 and
    prove nothing. The angle is range-reduced first -- `(2n+1)*k` modulo
    `4N`, exact because `cos(pi*x)` has period 2 in `x` -- which holds the
    evaluation error at `7.6e-07` LSB; the worst tie margin over all 5456
    coefficients of the supported extents is `4.9e-03` LSB, so the guard sits
    160x above the error and 40x below the closest coefficient. The reduction
    changes no emitted Q15, Q31 or f32 coefficient.

    The f32 profile maps `tensor<Nxf32>` to `tensor<Nxf32>` over the same
    coefficient matrix with the export boundary removed. Both numeric
    attributes carry the same contract; there is no requantization after a
    floating-point reduction, so the second exists only to keep the
    operand shape uniform with the fixed profile:

    ```text
    a = x[0] * c[k][0]                       one rounded product
    per n in 1 .. N-1:  a = a + x[n]*c[k][n] the contract-indexed term
    X[k] = a
    ```

    That is the base graph. `off` denotes it; `fma` fuses each term after
    the first, whose product has no addend; `fast` admits both plus any
    rebuild of the row's additive tree over the same indexed terms. Every
    row is a designated reduction, so R applies, but the tensor path does
    not reach the horizontal rewrite and the lowering selects the fused
    chain, spending F only.

    `c[k][n]` is the binary32 narrowing of a binary64 cosine, which makes
    the build's libm part of the declared constant rather than a detail:
    the coefficient bits are pinned, not derived.
  }];
  let arguments = (ins AnyRankedTensor:$input,
                   Ondsp_NumericAttr:$input_numeric,
                   Ondsp_NumericAttr:$output_numeric,
                   OptionalAttr<Ondsp_RoundingModeAttr>:$product_rounding,
                   OptionalAttr<Ondsp_RoundingModeAttr>:$rounding);
  let results = (outs AnyRankedTensor:$result);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def Ondrix_MovingAverageOp : Ondrix_Op<"moving_average", [Pure]> {
  let summary = "Valid-boundary moving average with one exact rounding";
  let description = [{
    `ondrix.moving_average` computes the valid-boundary sliding mean of a
    static rank-1 signed-Q1.15 tensor with a static window `K` in `[2, 64]`:

    ```text
    y[n] = round_nearest_even(sum_{i=0}^{K-1} x[n + i] / K),  n = 0..N-K
    ```

    The window sum is exact in i64 and the single explicit boundary is the
    final nearest-even division by `K`: `ondsp.round_shift` by `log2(K)`
    when `K` is a power of two (keeping that profile's lowering
    byte-identical to its pre-generalization form) and `ondsp.round_div`
    by `K` otherwise — the two spellings agree bit-exactly where both
    exist. The width costs this contract no new boundary: at most 64
    Q1.(W-1) values sum below `2^(W+6)`, exact in i64 at either width, and
    a mean of Q1.(W-1) values stays inside Q1.(W-1), so the declared
    saturation is unreachable either way. This contract is deliberately
    NOT the equal-tap FIR `sum x[n+i] * q15(1/K)`: quantizing `1/K` first
    is a different program with a different result (a gain-merge-class
    divergence), which is why the operation exists instead of a rewrite
    onto `ondrix.fir`.

    The f32 profile maps `tensor<Nxf32>` to `tensor<N-K+1xf32>` over the
    same index relation with both boundaries removed:

    ```text
    s = x[n]
    per i in 1 .. K-1:  s = s + x[n + i]     left to right, one IEEE add
    y[n] = s / f32(K)                        one IEEE division
    ```

    That window sum is a designated reduction, so `fast` does admit
    rebuilding it: the legal set holds more than one tree from `K = 2`, but
    a two-operand IEEE addition is bitwise commutative for every non-NaN
    result. At `K = 2` any remaining difference is confined to the NaN
    result's representation or which operand it comes from, neither of which
    the contract specifies; the invalid condition is symmetric in the
    operands, so exchange does not change it. The first difference in value
    appears at `K = 3`. The lowering builds the declared tree; a `fast`
    site additionally carries its declaration forward (an `ondsp.numeric`
    stamp on the window loop) so the batched schedule may rebuild the sum
    into independent chains, spending `rebuild_reduction_tree` once on the
    top fold. There is no product anywhere, so no contract can fuse, and
    `off` and `fma` denote the same events at every window the lowering
    emits. `K`
    is exact in binary32 over the admitted range, so the division is one
    correctly rounded event and carries no rounding attribute.
  }];
  let arguments = (ins AnyRankedTensor:$input,
                   I64Attr:$window,
                   Ondsp_NumericAttr:$numeric);
  let results = (outs AnyRankedTensor:$result);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

class Ondrix_TrigOp<string mnemonic> : Ondrix_Op<mnemonic, [Pure]> {
  let arguments = (ins AnyRankedTensor:$input,
                   Ondsp_NumericAttr:$numeric,
                   Ondsp_RoundingModeAttr:$rounding);
  let results = (outs AnyRankedTensor:$result);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def Ondrix_SineOp : Ondrix_TrigOp<"sine"> {
  let summary = "Fixed-point sine over uniform-Q turn phases, table plus correction";
  let description = [{
    `ondrix.sine` maps a static rank-1 tensor of `Q0.W` phases (`N` in
    `[1, 4096]`, `W` of 16 or 32) elementwise to `Q1.(W-1)` sines. A raw
    phase `p` is read as the unsigned `W`-bit value `u = p mod 2^W` covering
    one full turn (`angle = 2*pi*u/2^W`), so two's-complement wraparound IS
    phase wraparound and phase arithmetic stays exact.

    The phase width and the value width are the same `W`, and that is a
    contract rather than a convenience. Sine's derivative is `2*pi`, so a
    `Q0.16` phase quantizes the angle finely enough for about fourteen bits
    of value and no finer, no matter how large the table is. A wider value
    needs a wider angle.

    ### The Q15 profile: table plus linear interpolation

    ```text
    i      = u >> 8              256-entry table index
    f      = u & 255             Q8 interpolation fraction
    t[k]   = q15(sin(2*pi*k/256))    tie-guarded compile-time table
    y      = sat_i16(t[i] + round_half_even((t[(i+1) mod 256] - t[i]) * f / 2^8))
    ```

    One product boundary (the nearest-even interpolation rounding) and one
    declared saturating combine per element; the table itself carries the
    declared +1.0 saturation at the quarter turn and the exact -1.0 at
    three quarters. The whole 65536-phase input domain is finite, so the
    contract is gated exhaustively. Inadmissible tables fail closed in
    lowering (the committed 256-entry profile is admissible with a
    0.0044 LSB worst margin).

    ### The Q31 profile: table plus third-order angle addition

    Widening the shape above would not have widened the VALUE: a 256-entry
    table read by linear interpolation sits about `2^-15.7` from the real
    sine, so its storage can widen while its accuracy does not. Reaching
    `Q31` by that route needs a table of about `2^18` entries. The Q31
    profile is therefore a different construction on the same declared
    footing — one table, one rounding boundary, one saturating combine:

    ```text
    c      = u >> 22                   1024-entry table index
    r      = u & (2^22 - 1)            residual, an exact 2^-32 turn fraction
    t[k]   = q31(sin(2*pi*k/1024))     tie-guarded compile-time table
    S      = t[c]
    C      = t[(c + 256) mod 1024]     cos of a table angle is sin a quarter turn on
    A      = (round(2*pi * 2^38) * r) >> 38          the residual angle at 2^-32
    A2     = (A*A) >> 32,   A3 = (A2*A) >> 32
    D      = ((C*A) >> 1) - ((S*A2) >> 2) - ((C*A3) / 12)
    y      = sat_i32(S + round_half_even(D / 2^31))
    ```

    `D` is the angle-addition correction
    `S*(cos A - 1) + C*sin A` carried at `2^-62`, with
    `cos A ~ 1 - A^2/2` and `sin A ~ A - A^3/6`. The dropped `A^4/24` term is
    below `2^-33` because `A < 2*pi/1024`, which is what fixes the table at
    1024 entries rather than a size chosen for its own sake. One table
    serves both terms, so the constant is 4 KiB rather than 8. The `2*pi`
    scaling is the largest whose product with a 22-bit residual stays inside
    `i64`, and its `1.7e-13` relative error is far under the value's own LSB.

    Distance to the correctly-rounded sine, measured offline over ALL `2^32`
    phases: exact at 3022581719 of them, one LSB at 1272384953, two LSB at
    624, and never more. The two-LSB bound is what the execution gate
    enforces continuously, alongside bit-exactness against an independent
    restatement of the contract above.
  }];
}

def Ondrix_CosineOp : Ondrix_TrigOp<"cosine"> {
  let summary = "Fixed-point cosine over uniform-Q turn phases, table plus correction";
  let description = [{
    `ondrix.cosine` is `ondrix.sine` evaluated at `u + 2^(W-2) mod 2^W` (the
    exact quarter-turn phase advance, `16384` at `Q0.16` and `2^30` at
    `Q0.32`): identical table, index and correction contract, and
    boundaries — see `ondrix.sine`. The two operations therefore agree
    bit-exactly under the phase identity `cos(u) = sin(u + quarter turn)`,
    which the gate checks at both widths.
  }];
}

// The two fixed-point transcendentals that are inverses of each other, and
// the reason they can be exact contracts at all: the operation is not
// "approximately log" but "exactly this declared table with this declared
// interpolation", so its result is a bit-determined function of its input
// and the distance to the mathematical function is a documented property
// rather than a looseness in the contract.
class Ondrix_ExponentialOp<string mnemonic> : Ondrix_Op<mnemonic, [Pure]> {
  let arguments = (ins AnyRankedTensor:$input,
                   Ondsp_FixedAttr:$numeric,
                   Ondsp_FixedAttr:$output_numeric,
                   Ondsp_RoundingModeAttr:$rounding);
  let results = (outs AnyRankedTensor:$result);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def Ondrix_Log2Op : Ondrix_ExponentialOp<"log2"> {
  let summary = "Fixed-point base-two logarithm, exponent plus a mantissa table";
  let description = [{
    `ondrix.log2` maps a static rank-1 tensor (`N` in `[1, 4096]`)
    elementwise, at one of two widths. The raw i16 input is read as the
    UNSIGNED `Q0.16` magnitude `u / 2^16` in `[0, 1)`, and the result reads
    as signed `Q5.11`, which holds the whole output range `[-16, 0)` exactly;
    the raw i32 input is read as the unsigned `Q0.32` magnitude and the
    result as signed `Q6.26`, holding `[-32, 0]`. Both readings are in the
    attributes; neither is a scaling.

    ### The Q15 profile: exponent plus a 129-entry interpolated table

    ```text
    u == 0   -> -32768                    the declared value at the pole
    e        = highest set bit of u                       0 .. 15
    m        = u << (15 - e)              mantissa, in [2^15, 2^16)
    i        = (m >> 8) & 127             129-entry table index
    f        = m & 255                    Q8 interpolation fraction
    t[k]     = q11(log2(1 + k/128))       tie-guarded, t[128] = 2048 exactly
    y        = (e - 16) * 2048 + t[i] + round((t[i+1] - t[i]) * f / 2^8)
    ```

    One rounding boundary per element, at the interpolation; the exponent
    term is exact and its addition cannot leave the destination: the sum
    ranges over exactly `[-32768, 0]`, reaching `0` at the top input because
    the mantissa term reaches exactly one there (at `u = 65535` the
    interpolation rounds up to `t[128] = 2048`). The whole
    65536-value input domain is finite, so the contract is gated
    exhaustively rather than sampled. An input of zero has no logarithm and
    the operation does not pretend otherwise: `-32768` is the declared floor
    of the output format, named here rather than produced by a saturation
    the caller cannot see.

    ### The Q31 profile: exponent plus a table plus a residual series

    Widening the table above would not widen the value: a 129-point table
    read by linear interpolation is about `2^-16` from the real logarithm at
    any storage width. The Q31 profile is a different construction on the
    same footing — one table, one rounding boundary, no saturation:

    ```text
    u == 0   -> -2^31                     the declared value at the pole
    e        = highest set bit of u                       0 .. 31
    m        = u << (31 - e)              mantissa, in [2^31, 2^32)
    k        = m >> 21                    1024-entry table index, k - 1024
    d        = m - (k << 21)              residual below the table point
    r        = (d << 32) / (k << 21)      the residual RATIO at 2^-32, < 2^22
    t[j]     = q30(log2(1 + j/1024))      tie-guarded compile-time table
    series   = r * round(2^24/ln2) - ((r*r) >> 22) * round(2^13/ln2)
    y        = ((e - 32) << 26) + round_half_even((t[k-1024] << 26 + series) / 2^30)
    ```

    `series` is `log2(1 + r) ~ (r - r^2/2) / ln 2` carried at `2^56`; the
    dropped `r^3/3` term is below `2^-31.6` of a unit because `r < 2^-10`,
    which is what fixes the table at 1024 entries. The exponent term is
    exact and the sum ranges over exactly `[-2^31, 0]`, so nothing
    saturates. Distance to the correctly rounded logarithm, measured offline
    over ALL `2^32 - 1` nonzero inputs: exact at 4207881748 of them, one
    `Q6.26` LSB at 87085547, never more. The one-LSB bound is what the gate
    enforces against an independent `long double` reference, alongside
    bit-exactness against a restatement of the program above.
  }];
}

def Ondrix_Exp2Op : Ondrix_ExponentialOp<"exp2"> {
  let summary = "Fixed-point base-two exponential, the declared inverse of log2";
  let description = [{
    `ondrix.exp2` reverses `ondrix.log2`'s readings at either width: the
    input is signed `Q5.11` (`Q6.26`) and the result is the unsigned `Q0.16`
    (`Q0.32`) magnitude. Inputs at or above zero have no magnitude result
    and produce the declared ceiling, `65535` or `2^32 - 1`.

    ### The Q15 profile: two boundaries

    ```text
    v >= 0   -> 65535                     the declared value above the range
    e        = v >> 11                    exponent, -16 .. -1
    p        = v & 2047                   fractional part, Q11
    i        = p >> 4                     129-entry table index
    g        = p & 15                     Q4 interpolation fraction
    m[k]     = q15(2^(k/128))             tie-guarded, m[128] = 65536 exactly
    mantissa = m[i] + round((m[i+1] - m[i]) * g / 2^4)
    y        = round(mantissa / 2^(-e-1))
    ```

    Two rounding boundaries per element under one declared tie rule: the
    mantissa interpolation and the binade placement. They are separate
    events and the contract says so rather than folding them, because the
    second one is a shift by an input-dependent amount and no single rounding
    of the exact product would denote the same function.

    `exp2(log2(x))` is not the identity and the contract does not claim it
    is: each direction rounds once or twice, and the round trip is gated as
    a bounded discrepancy over the whole input domain rather than as an
    equality.

    ### The Q31 profile: one boundary

    ```text
    v >= 0   -> 2^32 - 1                  the declared value above the range
    e        = v >> 26                    exponent, -32 .. -1
    p        = v & (2^26 - 1)             fractional part, Q26
    k        = p >> 16                    1024-entry table index
    q        = p & 0xFFFF                 residual x = q / 2^26, below 2^-10
    m[j]     = 2^(j/1024) at 2^31, tie-guarded, read back unsigned
    a        = (q * round(ln2 * 2^40)) >> 26       x ln 2 at 2^40
    s        = a + (a*a >> 40)/2 + (a*a*a >> 80)/6  2^x - 1 at 2^40
    wide     = (m[k] << 31) + ((m[k] * s) >> 9)     the mantissa at 2^62
    y        = round_half_even(wide / 2^(30 - e))
    ```

    The binade placement is taken on the `2^62`-scaled mantissa, so the
    whole element has ONE rounding boundary where the Q15 profile needs two;
    it is still written out rather than declared as a `round_shift`, because
    its amount depends on the input. The dropped fourth-order term of the
    series is below `2^-46` of a unit. Distance to the correctly rounded
    exponential, measured offline over ALL `2^31` in-range inputs: exact at
    2114335012 of them, one `Q0.32` LSB at 33148636, never more. The gate
    enforces that bound alongside bit-exactness against a restatement.
  }];
}

def Ondrix_GoertzelOp : Ondrix_Op<"goertzel", [Pure]> {
  let summary = "Single-bin Goertzel energy with quantized recursion state";
  let description = [{
    `ondrix.goertzel` computes the squared magnitude of one DFT bin `k`
    of a static rank-1 signed-Q1.(W-1) tensor, for W of 16 or 32, of length
    `N` in `[2, 4096]` with `0 <= k <= N/2`, through the quantized Goertzel
    recursion:

    ```text
    c  = q15(cos(2*pi*k/N))       tie-guarded compile-time constant
    s1 = s2 = 0
    per sample n:
      m  = sat_i16(round_half_even(2*c*s1 / 2^15))
      s0 = sat_i16(x[n] + m - s2)
      s2 = s1; s1 = s0
    energy = s1*s1 + s2*s2 - sat_i16(round_half_even(2*c*s1 / 2^15)) * s2
    ```

    One product boundary and one saturating combine per sample are the
    observable state quantization. At Q15 the final energy is the exact i64
    value of the integer expression (`|energy| < 3 * 2^30`), returned as
    `tensor<1xi64>` with no further rounding.

    Q31 changes two things, and both are forced by the carrier rather than
    chosen. The step product is spelled `c*s1 / 2^(W-2)` instead of
    `2*c*s1 / 2^(W-1)` — the same rational, but the doubled form reaches
    `2^63` at Q31 while the folded one stays inside i64. And the energy is
    no longer exact: three terms of `2^62` leave i64, so each of `s1`, `s2`
    and `m` is requantized by ONE bit under a declared `state_rounding`
    before the squares, which puts the expression back below `2^62`. The
    returned reading is therefore two fractional bits coarser than the Q15
    one. `state_rounding` is required exactly at Q31 and refused at Q15,
    where the boundary does not exist.

    The coefficient is admissible only under the width's quantization tie
    guard; other bins fail closed in lowering. At Q31 that refusal is not
    theoretical the way it is at Q15: measured over all 4198399 admissible
    `(k, N)` sites, 1017 of them — 0.024%, one in 4130 — sit closer to a
    rounding tie than the `2^-13` LSB guard, so a binary64 evaluation
    cannot prove they round like the real-valued definition and the bin is
    refused by name. None of them is at `N <= 64`; the refusals begin where
    the coefficient space becomes dense enough to crowd the ties. Detecting one tone this way takes two words
    of recursion state instead of a full FFT's O(N) storage, at the cost
    of a different (also observable) quantization path — the Goertzel
    energy of a bin is NOT the squared magnitude of the corresponding
    staged-scaling FFT bin.

    The f32 profile maps `tensor<Nxf32>` to `tensor<1xf32>` over the same
    recursion with every boundary removed:

    ```text
    c2 = 2 * f32(cos(2*pi*k/N))   exact: doubling only moves the exponent
    s1 = s2 = +0.0
    per sample n:
      t  = x[n] + c2*s1           the one contract-indexed site
      s0 = t - s2
    energy = (s1*s1 + s2*s2) - (c2*s1)*s2
    ```

    Only the coefficient product and its input addition form a
    multiply-add, so that is the only site the contract indexes. The state
    subtraction and the closing energy expression are single IEEE operations
    in every mode, as with the mean of `ondrix.rms`.

    The base graph is that pseudocode read literally: one rounded product
    `c2*s1`, one addition of it to `x[n]`, single IEEE operations elsewhere.
    `off` denotes exactly it and `fma` the graph that fuses the single
    multiply-add site; both are exact contracts. `fast` denotes neither. No
    region here is a designated reduction: the sample chain is a data
    dependence, and the closing energy is a fixed expression rather than a sum
    over an index domain, so R has nothing to rebuild even though that
    expression is written with additive terms. The abstract legal set is those
    two graphs and nothing further. The lowering
    SELECTS the fused member and emits it carrying no permission of its own,
    which is what makes the result pinnable: a declaration left on the
    emitted event is not inert, because the backend may consume it and pick
    the other member. Selecting uniformly is this lowering's policy and not a
    rule of the language — the fusion choice is per term, and a schedule that
    mixes both is inside the same declared set. The f32 coefficient
    needs no tie guard either: the Q15 guard certifies a quantized table
    against an independently specified value, while here the binary32
    rounding IS the declared constant. Every admitted bin therefore lowers,
    where the Q15 profile fails closed on inadmissible ones. Both profiles
    evaluate the quarter-turn angles exactly rather than through libm, so
    bin `N/4` has the zero coefficient it names; the Q15 guard already
    quantized libm's 6.1e-17 to that zero, so this only moves the f32
    constant.
  }];
  let arguments = (ins AnyRankedTensor:$input,
                   I64Attr:$bin,
                   Ondsp_NumericAttr:$numeric,
                   OptionalAttr<Ondsp_RoundingModeAttr>:$state_rounding,
                   OptionalAttr<Ondsp_RoundingModeAttr>:$rounding);
  let results = (outs AnyRankedTensor:$energy);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def Ondrix_MatmulOp : Ondrix_Op<"matmul", [Pure]> {
  let summary = "Matrix product with one declared boundary per element";
  let description = [{
    `ondrix.matmul` multiplies static signed-Q1.15 matrices
    `tensor<MxKxi16> x tensor<KxNxi16> -> tensor<MxNxi16>` with all three
    dimensions in `[1, 64]`, per output element:

    ```text
    C[i][j] = sat_i16(round(sum_k A[i][k] * B[k][j] / 2^15))
    ```

    Every full product and the K-term sum are exact in i64
    (`|sum| <= 64 * 2^30 = 2^36`), so the single observable boundary per
    element is the final saturating requantization — the same contract
    shape as the FIR dot product, extended to a rank-2 result. The
    boundary's rounding is declared: `nearest_even` (the default),
    `toward_negative`, or `nearest_ties_positive`, the modes the export
    contract proves; each names a different function, never a hint.
    Saturation is declared and reachable (for example a full row of
    `-32768` against a matching column). Reassociating the exact K-sum is
    value-neutral by construction; moving or splitting the requantization
    boundary is not.

    The Q31 profile takes `tensor<MxKxi32> x tensor<KxNxi32>` to
    `tensor<MxNxi32>` over the same extent envelope. Its K-sum cannot be
    exact: one Q1.31 product already reaches `2^62`, so two of them leave
    i64. The profile therefore declares a SECOND boundary, on each product
    before it is accumulated:

    ```text
    t[k]    = round_shift(A[i][k] * B[k][j], p) under `product_rounding`
    C[i][j] = sat_i32(round(sum_k t[k] / 2^(31 - p)))
    ```

    `p = max(0, 2*(W-1) + floor(log2 K) - 62)` is the smallest shift that
    keeps the K-sum exact, derived from the inner extent exactly as the
    export shift is; it is not a parameter, and only its rounding is a
    choice. It is `floor` and not `ceil`: `K` terms each bounded by
    `2^(2*(W-1)-p)` sum below `2^(2*(W-1)-p+floor(log2 K)+1)`, so a `ceil`
    would spend one bit of precision at every non-power-of-two `K`. It is zero at every admitted Q15 shape and at `K = 1`, so
    `product_rounding` is required exactly where the boundary exists.

    That shift is also what bounds the narrowed K-sum: at `W = 32` it is
    `p = floor(log2 K)`, the `K <= 2^(p+1) - 1` terms are each at most
    `2^(62 - p)`, and `|sum| <= 2^63 - 2^(62 - p)` stays inside signed i64 —
    the counterpart of the `2^36` Q15 bound above. A wrapping i64 accumulator
    therefore never wraps here either, so both profiles reach the
    exact-modulo reassociation class without a prefix proof.

    The boundary is on the PRODUCT, not on the inputs, and that is a measured
    choice rather than a symmetry preference: narrowing an operand before the
    multiply scales its error by the OTHER operand's magnitude, so at `K = 64`
    an input-side narrowing is about `2^27` times less accurate than this one
    for the same sum bound. `ondrix.rms` narrows its input instead, and that
    is consistent rather than contradictory: its two factors are one value and
    its root compresses the error back, which this product does not have.

    The f32 profile maps `tensor<MxKxf32> x tensor<KxNxf32>` to
    `tensor<MxNxf32>` over the same extent envelope and carries no rounding
    attribute, because it has no requantization boundary to place: each
    element is exactly the reduction its `#ondsp.fp` contract declares over
    increasing `k`, from `+0.0`. Under `off` that is one multiply and one
    add per term, under `fma` one fused event per term, and under `fast` any
    schedule derivable from the declared rewrites.
  }];
  let arguments = (ins AnyRankedTensor:$lhs,
                   AnyRankedTensor:$rhs,
                   Ondsp_NumericAttr:$numeric,
                   OptionalAttr<Ondsp_RoundingModeAttr>:$product_rounding,
                   OptionalAttr<Ondsp_RoundingModeAttr>:$rounding);
  let results = (outs AnyRankedTensor:$result);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def Ondrix_RmsOp : Ondrix_Op<"rms", [Pure]> {
  let summary = "Root-mean-square with explicitly declared boundaries";
  let description = [{
    `ondrix.rms` reduces a static rank-1 signed-Q1.15 tensor with a
    power-of-two extent `N = 2^m` in `[2, 4096]` to its root-mean-square:

    ```text
    sumsq = sum x[i]^2                          exact i64, <= 2^42
    mean  = round_half_even(sumsq / 2^m)        first boundary, frac 30,
                                                nonnegative by construction
    rms   = sqrt_fixed(mean)                    second boundary, per the
                                                declared rounding, Q1.15
    ```

    The sum of squares is exact, so the only observable boundaries are the
    nearest-even mean and the integer square root. The mean is a sum of
    squares divided by a positive constant and therefore establishes the
    non-negative `ondsp.sqrt_fixed` domain structurally. The result is one
    element (`tensor<1xi16>`). The declared 32767 saturation of the root
    is reachable whenever the mean rounds to the top of the Q30 range:
    the all-minimum corner (mean exactly 2^30), but also mixed
    near-full-scale inputs — for example sixty-three minimum samples and
    one maximum, whose floor root 32767 takes the nearest adjustment to
    32768 before the clamp.

    The Q31 profile takes `tensor<Nxi32>` to `tensor<1xi32>` at the same
    power-of-two extents, and it cannot reuse the shape above: two squares of
    the Q1.31 minimum already sum to `2^63`, one past the signed i64 maximum,
    so an exact sum of squares overflows at EVERY admitted extent rather than
    at a large one. The profile therefore declares one more boundary than Q15
    does, before the products rather than after them:

    ```text
    x'    = round_shift(x, k) under `input_rounding`   pre-shift boundary
    sumsq = sum x'[i]^2                        exact i64, <= 2^(62 - 2k)
    mean  = round_half_even(sumsq / 2^m)       frac 2*(31 - k)
    rms   = sqrt_fixed(mean << 2k)             frac 31, the declared reading
    ```

    `k = ceil(m / 2)` is the smallest shift that makes the sum exact, and it
    is derived from the extent exactly as the mean's own shift is; it is not
    a parameter. Only its rounding is a choice, because a target's shifter
    decides it.

    The `2k` restoration is exact -- the mean is bounded by `2^(62 - 2k)`, so
    no bit leaves i64 -- and it is applied BEFORE the root rather than after
    it. Rooting the restored mean resolves the low `k` bits that
    `sqrt_fixed(mean) << k` would leave zero, so the result uses the full
    Q1.31 grid. Its ACCURACY is still what the pre-shift left: the input was
    truncated to `31 - k` bits, 25 of them at `N = 4096`. That is the price of
    the extent, and it is stated here rather than left to the caller.

    `k` is zero for every admitted Q15 extent and nonzero for every admitted
    Q31 one, so `input_rounding` is required exactly when the boundary exists
    and refused when it does not.

    The f32 profile takes `tensor<Nxf32>` to `tensor<1xf32>` for any `N` in
    `[2, 4096]` — the power-of-two restriction exists so the fixed-point
    mean is a shift, and an f32 mean is a division by a representable
    constant — and carries no rounding attribute:

    ```text
    sumsq = the reduction the #ondsp.fp contract declares over x[i]*x[i]
    mean  = sumsq / (f32)N          one IEEE division
    rms   = sqrt(mean)              one IEEE square root, correctly rounded
    ```

    Only the reduction is contract indexed. The division and the root are
    single operations with no grouping to choose and no product to fuse, so
    they are the same event under `off`, `fma`, and `fast` and carry no
    fast-math flag.
  }];
  let arguments = (ins AnyRankedTensor:$input,
                   Ondsp_NumericAttr:$numeric,
                   OptionalAttr<Ondsp_RoundingModeAttr>:$input_rounding,
                   OptionalAttr<Ondsp_RoundingModeAttr>:$rounding);
  let results = (outs AnyRankedTensor:$result);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def Ondrix_GainOp : Ondrix_Op<"gain", [Pure]> {
  let summary = "Elementwise constant Q15 gain with one explicit requantization";
  let description = [{
    `ondrix.gain` scales a static rank-1 signed-Q1.(W-1) tensor by one raw
    Q1.(W-1) constant `gain`, for W of 16 or 32, with per-element
    semantics:

    ```text
    y[i] = saturate_iW(round(x[i] * gain / 2^(W-1)))
    ```

    The product is exact in i64 at either width (`|x * gain| <= 2^(2W-2)`,
    so `2^30` at Q15 and `2^62` at Q31) and the single explicit boundary is
    the requantization back to Q1.(W-1); saturation is declared and
    reachable only for `x = gain = -2^(W-1)`. The constant is read at the
    declared width, so the same integer names a different real number at
    the two widths.

    Two tie rules are admissible at that boundary, and the choice is part
    of the contract: `nearest_even` (zero-biased) and
    `nearest_ties_positive` (add-half-then-shift, ties toward +infinity).
    They agree everywhere except on exact half-way products, and among
    those they differ exactly when the floor quotient is EVEN — which
    happens at both signs. With `gain = 3`, the raw input `-16384` gives
    the product `-49152 = -1.5 ulp` (floor quotient `-2`), requantized to
    `-2` under `nearest_even` and `-1` under `nearest_ties_positive`; a
    positive half whose floor quotient is `0` behaves the same way, giving
    `0` versus `1`.

    Cascade merging is a Q15-only rewrite. Its certificate is an exhaustive
    sweep of the 65536 Q15 inputs, and a Q31 sweep would be `2^32` inputs
    per candidate pair, which is not a compile-time check; `merge-ondrix-gain-cascades`
    therefore leaves Q31 cascades alone rather than merging them on a weaker
    argument.

    A cascade of two gain operations is in general a DIFFERENT program
    from one gain carrying the quantized product of the constants: with
    `g2 = 22938` then `g1 = 19661`, the cascade and the merged
    `q15(g1*g2/2^15) = 13763` differ on 10038 of the 65536 possible
    inputs, and the two cascade orders even differ from each other. The
    opt-in `merge-ondrix-gain-cascades` pass therefore rewrites a cascade
    only under an exhaustive compile-time certificate that evaluates both
    programs on all 65536 inputs and proves them bit-identical. That
    certificate is per tie rule: the mergeable constant pairs genuinely
    differ between `nearest_even` and `nearest_ties_positive`.

    The f32 profile scales by the declared `fp_gain` constant instead, with
    one IEEE multiply per element and no rounding attribute, because it has
    no requantization boundary to place a tie rule at. All three contract
    modes denote the same event graph here: a lone multiply offers neither a
    tree to rebuild nor an addend to fuse with, so the three declarations
    produce bit-identical results. The declaration remains at the call site;
    the emitted multiply carries no permission, because there is none to
    spend.
  }];
  let arguments = (ins AnyRankedTensor:$input,
                   OptionalAttr<I64Attr>:$gain,
                   OptionalAttr<F32Attr>:$fp_gain,
                   Ondsp_NumericAttr:$numeric,
                   OptionalAttr<Ondsp_RoundingModeAttr>:$rounding);
  let results = (outs AnyRankedTensor:$result);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def Ondrix_LmsOp : Ondrix_Op<"lms", [Pure]> {
  let summary = "Fixed-point LMS adaptive filter with quantized state";
  let description = [{
    `ondrix.lms` runs the least-mean-squares adaptation over a static
    rank-1 signed-Q1.15 input `x` (`N` in [1, 4096]) and desired signal
    `d`, starting from `K` initial weights (`K` in [1, 64]) with a raw
    Q1.15 step size `mu` in [0, 32767]. Prehistory is zero. Per sample
    `n`, in order:

    ```text
    acc     = sum_{k<K} w[k] * x[n-k]        exact i64 (x[m<0] = 0)
    y       = sat_i16(round_half_even(acc / 2^15))
    e[n]    = sat_i16(d[n] - y)
    step    = sat_i16(round_half_even(mu * e[n] / 2^15))
    w[k]   += round_half_even(step * x[n-k] / 2^15), saturated to i16
    ```

    Every sample takes `2 + K` observable rounding boundaries, and the
    updated weights feed the next sample: the quantized state is part of
    the contract. This is what makes the real-arithmetic rewrites of the
    update (fusing `mu * e * x` into one rounding, or folding an input
    gain into the step size) illegal — a one-LSB difference in one update
    compounds through the feedback. The operation returns the error
    signal and the final weights.

    The Q31 profile runs the same recursion over `tensor<Nxi32>` with a raw
    Q1.31 `mu` in `[0, 2^31 - 1]`. Exactly one of the five sites changes: the
    K-term tap sum, where one Q1.31 product already reaches `2^62` and two of
    them leave i64. Each tap product is therefore requantized by
    `p = floor(log2 K)` under a declared `product_rounding` before it joins the
    sum -- the same smallest-shift derivation `ondrix.matmul` states -- and
    the output boundary shifts by `31 - p`:

    ```text
    acc   = sum_k round_shift(w[k] * x[n-k], p)     exact i64
    y     = sat_i32(round(acc / 2^(31 - p)))
    ```

    The step and the weight update do NOT gain a boundary: each is a single
    product, which i64 still holds exactly. `p` is zero at every admitted Q15
    shape and at `K = 1`, so `product_rounding` exists exactly where the
    boundary does. Every sample therefore takes `2 + K` boundaries at Q15 and
    `2 + K + K` at Q31, and the quantized state remains part of the contract:
    the extra per-tap rounding compounds through the same feedback the Q15
    profile describes, which is why it is declared rather than derived at each
    call site.

    The f32 profile keeps the same recursion with the boundaries removed and
    the step size declared as `fp_step_size`:

    ```text
    y     = reduction_k w[k] * x[n-k]     contract indexed, from +0.0
    e[n]  = d[n] - y                      one IEEE subtract
    s     = mu * e[n]                     one IEEE multiply
    w[k] += s * x[n-k]                    contract indexed, per k
    ```

    Two sites are contract indexed: the K-term output reduction and each
    weight update. The sample recursion is not a permission question at all -
    it is a data dependence, so no declaration reorders samples, and `fast`
    authorizes rebuilding the tap sum's tree only. The step multiply has no
    site for either permission — its product's only consumer is a fused
    multiplicand — so it spends nothing and carries nothing, while the tap
    reduction and the weight updates select fused events and spend F. Unlike
    interpolation's inserted zeros, the zero prehistory samples `x[m < 0]`
    are evaluated rather than skipped, matching the fixed profile's shape.
  }];
  let arguments = (ins AnyRankedTensor:$input,
                   AnyRankedTensor:$desired,
                   AnyRankedTensor:$weights,
                   OptionalAttr<I64Attr>:$step_size,
                   OptionalAttr<F32Attr>:$fp_step_size,
                   OptionalAttr<Ondsp_RoundingModeAttr>:$product_rounding,
                   Ondsp_NumericAttr:$numeric,
                   OptionalAttr<Ondsp_RoundingModeAttr>:$rounding);
  let results = (outs AnyRankedTensor:$error, AnyRankedTensor:$adapted);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def Ondrix_CxMagnitudeOp : Ondrix_Op<"cx_magnitude", [Pure]> {
  let summary = "Fixed-point magnitude of a packed Q15 complex tensor";
  let description = [{
    `ondrix.cx_magnitude` computes the elementwise magnitude of a rank-1
    packed signed-Q1.15 complex tensor. For each packed element with real
    part `re` and imaginary part `im` (signed Q1.15):

    ```
    s      = re*re + im*im            (exact, at most 2^31)
    result = sqrt_fixed(s)            per ondsp.sqrt_fixed with the
                                      declared rounding, saturated to 32767
    ```

    The sum of squares is an exact integer carrying fraction 30, so the
    integer square root is the exact Q1.15 magnitude before the declared
    rounding; there is no intermediate requantization. Saturation is
    declared and reachable only for magnitudes of at least +1.0 (for
    example `re = im = -32768`). Composed after `ondrix.rfft`, this yields
    the fixed-point magnitude spectrum of the staged-scaling contract.

    The Q31 profile reads `packed_i32_imag_hi_real_lo` -- the layout the Q31
    transforms already freeze -- taking `tensor<Nxi64>` to `tensor<Nxi32>`.
    Its sum of squares is NOT exact: two squares of the Q1.31 minimum reach
    exactly `2^63`, one past the signed i64 maximum, so the profile declares a
    pre-shift on each component before it is squared:

    ```
    re'    = round_shift(re, 1) under `input_rounding`
    im'    = round_shift(im, 1) likewise
    s      = re'*re' + im'*im'        exact, at most 2^61
    result = sqrt_fixed(s << 2)       frac 31, the declared reading
    ```

    The shift is one bit, which is what `2*(31 - k) + 1 <= 62` forces, and it
    is derived rather than declared -- only its rounding is a choice. The
    boundary is on the components rather than on each square for the same
    measured reason `ondrix.rms` places it there: this operation ends in a
    square root, which compresses the error a linear reduction would keep.

    The Q15 executable profile is limited to static rank-1 `tensor<Nxi32>`
    packed input to `tensor<Nxi16>` output with `N` in `[1, 4096]`, the
    imag-high/real-low packed layout, signed Q15, and `toward_negative` or
    `nearest_even` rounding (the integer square root has no reachable
    nearest ties). `input_rounding` is refused there: an exact sum of two
    Q15 squares reaches only `2^31`.
  }];
  let arguments = (ins AnyRankedTensor:$input,
                   Ondsp_CxLayoutAttr:$layout,
                   Ondsp_NumericAttr:$numeric,
                   OptionalAttr<Ondsp_RoundingModeAttr>:$input_rounding,
                   Ondsp_RoundingModeAttr:$rounding);
  let results = (outs AnyRankedTensor:$result);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def Ondrix_CxPhaseOp : Ondrix_Op<"cx_phase", [Pure]> {
  let summary = "Fixed-point phase of a packed complex tensor, as a turn";
  let description = [{
    `ondrix.cx_phase` maps each packed signed-Q1.(W-1) complex element, `W`
    of 16 or 32, to its argument, expressed in the unsigned turn
    `ondrix.sine` reads: the result is `atan2(im, re) / (2*pi) * 2^T` taken
    modulo `2^T`, so `sine(cx_phase(z))` is a well-formed program and phase
    arithmetic stays exact under two's-complement wraparound. The turn width
    `T`, 16 or 32, is declared in its own attribute and independently of the
    component width: the `Q0.16` turn is what the Q15 sine reads and the
    `Q0.32` turn what the Q31 sine reads, and either component width can
    produce either, because the components only ever enter through the
    ratio `r` and the ratio is carried to the turn's precision.

    Only the absolute values and the ratio numerator follow the component
    width, and they run one bit wider than the component, because negating
    the component minimum is part of taking an absolute value.

    ### The Q0.16 turn: a 129-entry interpolated table

    ```text
    (re, im) == (0, 0)  ->  0                the declared value at the origin
    a, b   = |re|, |im|                      exact in i32
    hi, lo = max(a,b), min(a,b)
    r      = round(lo * 2^16 / hi)           in [0, 65536]
    i      = min(r >> 9, 127)                129-entry table index
    f      = r - i * 512                     Q9 interpolation fraction
    t[k]   = q16(atan(k/128) / (2*pi))       tie-guarded, t[128] = 8192 exactly
    base   = t[i] + round((t[i+1] - t[i]) * f / 2^9)
    q      = b > a ? 16384 - base : base     the octant fold, exact
    turn   = re >= 0 ? (im >= 0 ? q : -q) : (im >= 0 ? 32768 - q : 32768 + q)
    ```

    Two rounding boundaries per element, the ratio and the interpolation,
    both under the one declared tie rule. Everything else — the absolute
    values, the octant fold, and the quadrant unfold — is exact index
    arithmetic on the turn, which is why the eight octants meet exactly at
    their boundaries instead of nearly: the axes give `0`, `16384`, `32768`
    and `49152`, and the diagonals give the odd multiples of `8192`.

    The input domain is `2^32` pairs, so unlike `ondrix.sine` and
    `ondrix.log2` this contract is NOT gated exhaustively, and the gate says
    which structure it covers instead: every value of one component against
    fixed values of the other, all four axes and all four diagonals over
    their whole range, the rails, and a deterministic sweep. That is a
    weaker evidence class than an exhaustive one and is recorded as such.

    ### The Q0.32 turn: exact angle addition over a 1025-entry table

    Widening the table above would not widen the turn — a 129-point table
    read by linear interpolation is about `2^-16` from the real arctangent
    at any storage width — so the wide turn is a different construction:

    ```text
    R      = round(lo * 2^32 / hi)           the ratio at 2^-32, in [0, 2^32]
    k      = R >> 22                         1025-entry table index, 0..1024
    d      = R - (k << 22)                   residual below the table point
    u      = (d << 42) / (2^42 + R*k)        (x - r_k)/(1 + x*r_k) at 2^-32
    t[j]   = atan(j/1024) / (2*pi) at 2^33, tie-guarded, t[1024] = 2^30 exactly
    corr   = (u*K) >> 7 - (((u^3 >> 44) * K) / 3) >> 27,   K = round(2^40/(2*pi))
    base   = round_half_even((t[k] << 32 + corr) / 2^33)   in [0, 2^29]
    q      = b > a ? 2^30 - base : base
    turn   = re >= 0 ? (im >= 0 ? q : -q) : (im >= 0 ? 2^31 - q : 2^31 + q)  mod 2^32
    ```

    `atan(x) = atan(r_k) + atan(u)` is exact, and `u < 2^-10` so
    `atan(u) ~ u - u^3/3` drops a term below `2^-52`. Two rounding
    boundaries per element again, the ratio and the return to `Q0.32`, both
    nearest-even; the second division truncates and its `2^-32` error is
    under a twentieth of an output LSB. The `2^64` input domain cannot be
    swept; measured over `2^30` pseudo-random Q31 pairs the result is exact
    at 918798647 of them, one `Q0.32` LSB from the correctly rounded turn at
    154943177, never more, and the gate enforces that one-LSB bound at both
    component widths alongside bit-exactness against a restatement. The axes
    and diagonals are exact at both turn widths.
  }];
  let arguments = (ins AnyRankedTensor:$input,
                   Ondsp_CxLayoutAttr:$layout,
                   Ondsp_NumericAttr:$numeric,
                   Ondsp_FixedAttr:$output_numeric,
                   Ondsp_RoundingModeAttr:$rounding);
  let results = (outs AnyRankedTensor:$result);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def Ondrix_WindowHammingOp : Ondrix_Op<"window_hamming", [Pure]> {
  let summary = "Compile-time symmetric Hamming window coefficients";
  let description = [{
    `ondrix.window_hamming` names the symmetric Hamming window of static
    length `N` taken from the result type:

    ```text
    w[n] = 27/50 - (23/50) * cos(2*pi*n / (N-1)),  n = 0..N-1
    ```

    Each real coefficient is quantized once to signed Q1.(W-1), for W of 16
    or 32, with round-half-even and declared saturation at the +1.0 window
    center. The window value itself is width-free, so the width moves the
    quantization boundary and nothing else. The operation is a compile-time
    design intent with no runtime form: the opt-in
    `evaluate-ondrix-fir-design` pass replaces it with a constant
    coefficient tensor, and generation fails closed when any coefficient is
    closer to a rounding tie than the guard for its width, so the emitted
    table provably equals the real-valued definition and is exactly
    symmetric.

    At Q31 that refusal is common enough that the rate belongs in the
    contract. Swept over every extent, the `2^-13` LSB guard refuses 20.1%
    of Hamming extents, 15.9% of Hann and 22.4% of Blackman; Kaiser runs
    1.2% to 5.1% depending on beta. No extent below 225 is refused for any
    of the three cosine windows, so the short designs are unaffected. The
    Q15 figures are 0.20%, 0.17% and 0.07%. The difference is the guard and
    not the arithmetic -- binary64 evaluates these windows to about
    `3e-06` Q31 LSB, forty times inside the guard -- and the guard is one
    cross-operation contract rather than this operation's to loosen.

    The result is a static rank-1 `tensor<NxiW>` with `2 <= N <= 4096` and a
    matching uniform-Q `numeric`.
  }];
  let arguments = (ins Ondsp_NumericAttr:$numeric);
  let results = (outs AnyRankedTensor:$coefficients);
  let hasVerifier = 1;
  let assemblyFormat = "attr-dict `:` type($coefficients)";
}

def Ondrix_WindowHannOp : Ondrix_Op<"window_hann", [Pure]> {
  let summary = "Compile-time symmetric Hann window coefficients";
  let description = [{
    `ondrix.window_hann` names the symmetric Hann window of static length
    `N` taken from the result type:

    ```text
    w[n] = 1/2 - (1/2) * cos(2*pi*n / (N-1)),  n = 0..N-1
    ```

    The endpoints are exactly zero and the +1.0 center of odd lengths
    saturates to the format maximum by declared convention. Quantization,
    the width choice, the fail-closed tie guard and its measured Q31
    refusal rate, symmetry, and the `evaluate-ondrix-fir-design` consumer
    all follow `ondrix.window_hamming`.

    The result is a static rank-1 `tensor<NxiW>` with `2 <= N <= 4096` and a
    matching uniform-Q `numeric`.
  }];
  let arguments = (ins Ondsp_NumericAttr:$numeric);
  let results = (outs AnyRankedTensor:$coefficients);
  let hasVerifier = 1;
  let assemblyFormat = "attr-dict `:` type($coefficients)";
}

def Ondrix_WindowBlackmanOp : Ondrix_Op<"window_blackman", [Pure]> {
  let summary = "Compile-time symmetric Blackman window coefficients";
  let description = [{
    `ondrix.window_blackman` names the classic symmetric Blackman window
    (alpha = 4/25) of static length `N` taken from the result type:

    ```text
    w[n] = 21/50 - (1/2) * cos(2*pi*n / (N-1))
                 + (2/25) * cos(4*pi*n / (N-1)),  n = 0..N-1
    ```

    The endpoints are exactly zero (21/50 - 1/2 + 2/25 = 0). Quantization,
    the width choice, the fail-closed tie guard and its measured Q31
    refusal rate, symmetry, and the `evaluate-ondrix-fir-design` consumer
    all follow `ondrix.window_hamming`.

    The result is a static rank-1 `tensor<NxiW>` with `2 <= N <= 4096` and a
    matching uniform-Q `numeric`.
  }];
  let arguments = (ins Ondsp_NumericAttr:$numeric);
  let results = (outs AnyRankedTensor:$coefficients);
  let hasVerifier = 1;
  let assemblyFormat = "attr-dict `:` type($coefficients)";
}

def Ondrix_WindowKaiserOp : Ondrix_Op<"window_kaiser", [Pure]> {
  let summary = "Compile-time symmetric Kaiser window coefficients";
  let description = [{
    `ondrix.window_kaiser` names the symmetric Kaiser window of static
    length `N` taken from the result type, with rational shape parameter
    `beta = beta_num / beta_den` in `(0, 50]`:

    ```text
    r[n] = (2*n - (N-1)) / (N-1)
    w[n] = I0(beta * sqrt(1 - r[n]^2)) / I0(beta)
    ```

    `I0` is the zeroth-order modified Bessel function of the first kind,
    evaluated at compile time by its power series to a documented
    binary64 error budget; the fail-closed quantization tie guard for the
    declared width backstops the whole evaluation chain exactly as for the
    other window designs. Its Q31 refusal rate is the lowest of the four,
    1.2% to 5.1% of extents depending on beta, because the series terms are
    all positive and never cancel. The endpoints are `1 / I0(beta)`
    (nonzero) and the +1.0 center of odd lengths saturates to the format
    maximum by declared convention.

    The result is a static rank-1 `tensor<NxiW>` with `2 <= N <= 4096`
    and a matching uniform-Q `numeric`.
  }];
  let arguments = (ins I64Attr:$beta_num,
                   I64Attr:$beta_den,
                   Ondsp_NumericAttr:$numeric);
  let results = (outs AnyRankedTensor:$coefficients);
  let hasVerifier = 1;
  let assemblyFormat = "attr-dict `:` type($coefficients)";
}

def Ondrix_FirDesignWindowedSincOp
    : Ondrix_Op<"fir_design_windowed_sinc", [Pure]> {
  let summary = "Compile-time Hamming-windowed sinc FIR design coefficients";
  let description = [{
    `ondrix.fir_design_windowed_sinc` names a linear-phase FIR design of
    static odd length `N` (from the result type) obtained by truncating the
    ideal response at rational cutoff `fc = cutoff_num / cutoff_den`
    (relative to the sample rate, `0 < fc < 1/2` strictly) and applying the
    symmetric Hamming window `w` of `ondrix.window_hamming`. With
    `M = (N-1)/2` and `sinc(x) = sin(pi*x)/(pi*x)`, `sinc(0) = 1`:

    ```text
    lowpass:   h[n] = 2*fc * sinc(2*fc*(n-M)) * w[n]
    highpass:  h[n] = delta[n-M] - lowpass[n]     (spectral inversion)
    ```

    Each composite real coefficient is quantized exactly once to signed
    Q1.(W-1), for W of 16 or 32, with round-half-even and declared
    saturation; there is no quantize-then-combine step. The sine argument is
    reduced exactly first -- `2*cutoff_num*(n-M)` modulo `2*cutoff_den`,
    exact because `sin(pi*x)` has period 2 in `x`, and applied to the
    NUMERATOR only because the denominator `pi*x` is the value itself.
    Without it the argument reaches `pi*2047` at the maximum extent and its
    binary64 error becomes about `3e-03` Q31 LSB, past any guard these
    coefficients could survive. The operation is a compile-time design
    intent evaluated by the opt-in `evaluate-ondrix-fir-design` pass, which
    fails closed when any coefficient is closer to a rounding tie than the
    guard for its width. At Q31 that refusal is reachable rather than
    theoretical: measured tie margins for representative designs run from
    `2.4e-04` to `1.4e-02` LSB against a `2^-13` guard, so the closest of
    them clears it by under a factor of two and a longer design can fall
    inside. Under that guard the emitted integer table equals the
    real-valued definition and is exactly symmetric, so it feeds the
    existing constant-coefficient analysis, symmetric-pairing proofs, and
    proof-gated Vector paths unchanged.

    The result is a static rank-1 `tensor<NxiW>` with odd `3 <= N <= 4095`
    and a matching uniform-Q `numeric`.
  }];
  let arguments = (ins Ondrix_FirDesignResponseAttr:$response,
                   I64Attr:$cutoff_num, I64Attr:$cutoff_den,
                   Ondsp_NumericAttr:$numeric);
  let results = (outs AnyRankedTensor:$coefficients);
  let hasVerifier = 1;
  let assemblyFormat = "attr-dict `:` type($coefficients)";
}

// The elementwise family. Every member maps a static rank-1 uniform-Q
// `tensor<NxiW>` (`W` of 16 or 32, `N` in [1, 4096]) elementwise and states
// the same thing about itself: what the exact result of the operation is
// over the integers, and what the declared boundary does to it. The
// exactness claim is what makes the boundary the only place a value can
// change, so these operations compose without a rewrite having to
// reconstruct where rounding happened. The exact body therefore runs at
// twice the storage, and the two members carrying a raw attribute take the
// range the declared format actually has rather than a pinned one.
class Ondrix_ElementwiseOp<string mnemonic> : Ondrix_Op<mnemonic, [Pure]> {
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
  let results = (outs AnyRankedTensor:$result);
}

class Ondrix_ExactCombineOp<string mnemonic, string noun, string symbol>
    : Ondrix_ElementwiseOp<mnemonic> {
  let summary = "Elementwise uniform-Q " # noun # " with one declared overflow";
  let description = [{
    ```text
    y[i] = narrow_iW(x[i] }] # symbol # [{ z[i])    the sum is exact one bit wider
    ```

    No information is lost below the least significant bit, so there is no
    rounding boundary and none is declared. The }] # noun # [{ leaves
    `Q1.(W-1)` exactly when the operands have the same sign and enough
    magnitude, which the declared `overflow` resolves: `saturate` clamps to
    the rail and `wrap` returns the exact value modulo `2^W`.
  }];
  let arguments = (ins AnyRankedTensor:$lhs,
                   AnyRankedTensor:$rhs,
                   Ondsp_NumericAttr:$numeric,
                   Ondsp_OverflowModeAttr:$overflow);
}

def Ondrix_AddOp : Ondrix_ExactCombineOp<"add", "sum", "+">;
def Ondrix_SubOp : Ondrix_ExactCombineOp<"sub", "difference", "-">;

def Ondrix_MultOp : Ondrix_ElementwiseOp<"mult"> {
  let summary = "Elementwise uniform-Q product with one explicit requantization";
  let description = [{
    ```text
    y[i] = narrow_iW(round(x[i] * z[i] / 2^(W-1)))    the product is exact
    ```

    This is `ondrix.gain` with a tensor in place of the constant, and it has
    the same two boundaries in the same order: the exact `Q(2W-2)` product
    loses `W-1` bits under the declared tie rule, and the result is narrowed
    under the declared overflow. The product needs a `2W`-bit carrier to be
    exact, which is what the lowering uses. Saturation is reachable only at
    the two storage minima, where the product is exactly `+1.0`.

    All four declared tie rules are admissible, and they genuinely disagree:
    the rules split on halfway products, and among those `nearest_even` and
    `nearest_ties_positive` split further on the parity of the floor
    quotient. Unlike the constant form there is no compile-time certificate
    to run, because neither operand is known, which is precisely why a
    cascade of two `mult` operations has no merged form at all.
  }];
  let arguments = (ins AnyRankedTensor:$lhs,
                   AnyRankedTensor:$rhs,
                   Ondsp_NumericAttr:$numeric,
                   Ondsp_RoundingModeAttr:$rounding,
                   Ondsp_OverflowModeAttr:$overflow);
}

class Ondrix_ExactUnaryOp<string mnemonic, string noun, string formula>
    : Ondrix_ElementwiseOp<mnemonic> {
  let summary = "Elementwise uniform-Q " # noun # " with one declared overflow";
  let description = [{
    ```text
    y[i] = narrow_iW(}] # formula # [{)
    ```

    The }] # noun # [{ is exact over the integers and there is no rounding
    boundary. Its whole finite-precision content is one asymmetry: the signed
    `Q1.(W-1)` range is `[-2^(W-1), 2^(W-1)-1]`, so the storage minimum is
    the single input whose result is unrepresentable. Producing its magnitude
    at all needs a carrier wider than the storage. The declared `overflow`
    says what happens there — `saturate` gives the maximum and `wrap` gives
    the minimum back — and every other input is unaffected by the choice.
  }];
  let arguments = (ins AnyRankedTensor:$input,
                   Ondsp_NumericAttr:$numeric,
                   Ondsp_OverflowModeAttr:$overflow);
}

def Ondrix_AbsOp : Ondrix_ExactUnaryOp<"abs", "absolute value", "|x[i]|">;
def Ondrix_NegateOp : Ondrix_ExactUnaryOp<"negate", "negation", "-x[i]">;

def Ondrix_OffsetOp : Ondrix_ElementwiseOp<"offset"> {
  let summary = "Elementwise uniform-Q constant bias with one declared overflow";
  let description = [{
    ```text
    y[i] = narrow_iW(x[i] + bias)    the sum is exact one bit wider
    ```

    The constant form of `ondrix.add`, with the same exactness and the same
    single declared boundary. `bias` is a raw signed `Q1.(W-1)` value in
    `[-2^(W-1), 2^(W-1)-1]`; it is a raw value in the declared format, so
    the same integer names a different real number at the two widths.
  }];
  let arguments = (ins AnyRankedTensor:$input,
                   I64Attr:$bias,
                   Ondsp_NumericAttr:$numeric,
                   Ondsp_OverflowModeAttr:$overflow);
}

def Ondrix_ShiftOp : Ondrix_ElementwiseOp<"shift"> {
  let summary = "Elementwise uniform-Q binary scaling by a declared power of two";
  let description = [{
    ```text
    amount >= 0:  y[i] = narrow_iW(x[i] * 2^amount)          exact, no tie
    amount <  0:  y[i] = narrow_iW(round(x[i] / 2^-amount))  one tie rule
    ```

    `amount` is in `[-(W-1), W-1]`, so the left shift stays exact inside the
    `2W`-bit carrier. The two directions are not the same kind of
    operation and the contract does not pretend they are: a left shift can
    only overflow and a right shift can only round. That is why the rounding
    mode is declared even when it is vacuous — the sign of one attribute must
    not silently decide whether another one means anything, and a program
    that changes the amount keeps the tie rule it asked for.

    A right shift is the operation where the four tie rules are furthest
    apart, and it is how a target's rounding convention is identified: at
    `amount = -1` the inputs `1, 3, -1, -3` give `1, 2, 0, -1` under
    `nearest_ties_positive`, `0, 2, 0, -2` under `nearest_even`, and
    `0, 1, -1, -2` under `toward_negative`. Those four values separate the
    rules completely, which no single positive input does.
  }];
  let arguments = (ins AnyRankedTensor:$input,
                   I64Attr:$amount,
                   Ondsp_NumericAttr:$numeric,
                   Ondsp_RoundingModeAttr:$rounding,
                   Ondsp_OverflowModeAttr:$overflow);
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
