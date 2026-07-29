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
    contract is signed Q15 with an exact full product, a signed frac30
    accumulator of at least 32 bits, and explicit export policy. Input phase,
    padding, interpolation, cross-output polyphase reassociation, and target
    selection are outside this operation. Vectorization is not part of the
    mathematical contract; consumers may batch products only while preserving
    the declared ordered updates.

    Dynamic extent relationships are execution preconditions. Generic
    lowering emits dominating assertions before evaluating any window.
  }];
  let arguments = (ins
      AnyRankedTensor:$input,
      AnyRankedTensor:$coeffs,
      AnyRankedTensor:$init,
      I64Attr:$factor,
      Ondsp_FixedAttr:$numeric,
      Ondsp_ProductAttr:$product,
      TypeAttrOf<Ondsp_Acc>:$accumulator,
      Ondsp_FixedAttr:$dst,
      Ondsp_RoundingModeAttr:$rounding,
      Ondsp_OverflowModeAttr:$overflow);
  let results = (outs AnyRankedTensor:$result);
  let extraClassDeclaration = [{
    std::pair<int64_t, int64_t> getDpsInitsPositionRange() {
      return {2, 3};
    }
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

    Skipping inserted-zero terms is part of this fixed-point contract and does
    not authorize reassociation of the remaining updates. The `init` tensor
    provides destination shape and storage; its element values are not part
    of the mathematical result.

    The first executable profile accepts exactly `factor = 2`, signed Q15,
    exact full products, a signed frac30 accumulator of at least 32 bits, and
    explicit export policy. Stateful block forms, configurable phase,
    polyphase scheduling, floating point, Q31, Vector, and target selection
    are outside this operation.

    Dynamic extent relationships and index arithmetic are execution
    preconditions. Generic lowering emits dominating assertions before
    evaluating any sample.
  }];
  let arguments = (ins
      AnyRankedTensor:$input,
      AnyRankedTensor:$coeffs,
      AnyRankedTensor:$init,
      I64Attr:$factor,
      Ondsp_FixedAttr:$numeric,
      Ondsp_ProductAttr:$product,
      TypeAttrOf<Ondsp_Acc>:$accumulator,
      Ondsp_FixedAttr:$dst,
      Ondsp_RoundingModeAttr:$rounding,
      Ondsp_OverflowModeAttr:$overflow);
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
    not authorize reassociation of tap updates.
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

    `mul_add` follows the `numeric` contraction policy: `off` emits separate
    multiply and add operations, `fma` requires one fused multiply-add, and
    `fast` permits fast-math transformations. The standalone `mul` and `add`
    steps remain distinct operations except that `fast` attaches fast-math
    flags. Empty input returns an empty output and the unchanged state.

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
  let summary = "Static packed-Q15 radix-2 CFFT";
  let description = [{
    Computes one static power-of-two radix-2 decimation-in-time CFFT over
    packed signed-Q15 complex values, for extents `N` in `[4, 1024]`. The
    input and result use natural index order and matching `tensor<Nxi32>`
    storage. Each packed element uses the layout and finite-precision
    butterfly contract of `ondrix.butterfly`.

    The transform recursively decomposes even and odd input indices, then
    combines the two half transforms in increasing frequency order:

      `(y[k], y[k + N/2]) = butterfly(even[k], odd[k], W_N^k)`

    Forward direction uses `W_N^k = exp(-2*pi*j*k/N)`; inverse direction
    uses `conj(W_N^k)`. Each twiddle component is quantized once from the
    real-valued definition with round-half-even signed Q1.15 and declared
    saturation (`+1.0` and `+j` use positive raw value 32767, `-1.0` and
    `-j` use raw value -32768), and generation is admissible only when the
    binary64 estimate is at least `2^-20` Q15 LSB away from every rounding
    tie, so the emitted constants provably equal the real-valued definition
    under the declared binary64/libm evaluation error budget (more than
    three orders of magnitude below the guard). Four points use `[1, -j]`
    forward; eight
    points use `[1, 23170-j23170, -j, -23170-j23170]` forward, with the
    conjugate tables in inverse direction.

    Each butterfly applies the declared product and output requantization
    independently. A size-N transform therefore has `log2(N)` observable
    scaling boundaries; they must not be collapsed into one final
    real-arithmetic normalization. With the current one-bit output scale at
    every stage, inverse direction includes the conventional `1/N`
    normalization.

    The executable profile is deliberately limited to static power-of-two
    extents in `[4, 1024]`, explicit forward or inverse direction, the
    imag-high/real-low packed layout, signed Q15, exact full products, and
    the same two scale policies as the executable butterfly. Dynamic
    planning, other sizes, and other numeric profiles require separate
    extensions.
  }];
  let arguments = (ins AnyRankedTensor:$input,
                   Ondrix_CfftDirectionAttr:$direction,
                   Ondsp_CxLayoutAttr:$layout,
                   Ondsp_NumericAttr:$numeric,
                   Ondsp_ProductAttr:$product,
                   Ondsp_ScaleAttr:$product_scale,
                   Ondsp_ScaleAttr:$output_scale);
  let results = (outs AnyRankedTensor:$result);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def Ondrix_RfftOp : Ondrix_Op<"rfft", [Pure]> {
  let summary = "Static packed-Q15 real-to-Hermitian FFT";
  let description = [{
    Computes one static power-of-two real FFT, for extents `N` in
    `[8, 1024]`, from signed-Q15 `tensor<Nxi16>` input to compact
    natural-order `tensor<(N/2+1)xi32>` Hermitian bins. Each result uses the
    declared packed complex layout.

    The reference schedule embeds every real input `x[n]` as
    `x[n] + j*0`, evaluates the size-N staged radix-2 CFFT equation, and
    retains bins `0..N/2`. It uses the same tie-guarded generated Q15
    twiddles and the same product and output requantization boundaries as
    `ondrix.cfft`. For example, the size-16 forward twiddle pairs
    `(real, imaginary)` are:

    ```
    [(32767, 0), (30274, -12540), (23170, -23170), (12540, -30274),
     (0, -32768), (-12540, -30274), (-23170, -23170), (-30274, -12540)]
    ```

    The one-bit output scale at each of `log2(N)` stages has nominal cumulative
    scale `1/N`. The sequence of nearest-even saturating stage operations is
    the exact contract; it must not be collapsed into one final division.

    The imaginary components of the DC and Nyquist results are canonically
    zero. Half-size packing and real-pair recombination are optimizations of
    this contract, not part of the reference schedule.

    The executable profile is limited to static power-of-two N in
    `[8, 1024]`, the imag-high/real-low packed layout, signed Q15, exact
    full products, nearest-even saturating Q30-to-Q15 product scaling, and
    nearest-even saturating one-bit output scaling.
  }];
  let arguments = (ins AnyRankedTensor:$input,
                   Ondsp_CxLayoutAttr:$layout,
                   Ondsp_NumericAttr:$numeric,
                   Ondsp_ProductAttr:$product,
                   Ondsp_ScaleAttr:$product_scale,
                   Ondsp_ScaleAttr:$output_scale);
  let results = (outs AnyRankedTensor:$result);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def Ondrix_IrfftOp : Ondrix_Op<"irfft", [Pure]> {
  let summary = "Static packed-Q15 Hermitian-to-real inverse FFT";
  let description = [{
    Computes one static power-of-two inverse real FFT, for extents `N` in
    `[8, 1024]`, from compact natural-order `tensor<(N/2+1)xi32>` Hermitian
    bins to signed-Q15 `tensor<Nxi16>` output.

    The reference schedule canonicalizes the imaginary components of the DC
    and Nyquist bins to zero, copies bins `1..N/2-1`, and reconstructs bin
    `N-k` by signed-Q15 saturating conjugation of bin `k`. Saturating
    conjugation maps imaginary raw value -32768 to +32767. The reconstructed
    size-N spectrum is evaluated by the staged inverse CFFT equation, and the
    real component of each output is returned. Inverse twiddles use the same
    tie-guarded generation as `ondrix.cfft`; for example, the size-16
    inverse twiddle pairs `(real, imaginary)` are:

    ```
    [(32767, 0), (30274, 12540), (23170, 23170), (12540, 30274),
     (0, 32767), (-12540, 30274), (-23170, 23170), (-30274, 12540)]
    ```

    Every inverse CFFT product and output requantization boundary remains
    observable. The one-bit output scale at each stage has nominal cumulative
    scale `1/N`, but the stage operations must not be collapsed. Consequently,
    `irfft(rfft(x))` is the sequential finite-precision program and is not an
    identity rewrite.

    The executable profile is limited to static power-of-two N in
    `[8, 1024]`, the imag-high/real-low packed layout, signed Q15, exact
    full products, nearest-even saturating Q30-to-Q15 product scaling, and
    nearest-even saturating one-bit output scaling.
  }];
  let arguments = (ins AnyRankedTensor:$input,
                   Ondsp_CxLayoutAttr:$layout,
                   Ondsp_NumericAttr:$numeric,
                   Ondsp_ProductAttr:$product,
                   Ondsp_ScaleAttr:$product_scale,
                   Ondsp_ScaleAttr:$output_scale);
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
    exact in i64 — `|sum| <= N * 2^30 < 2^36` — so the reduction has no
    reachable intermediate overflow and its association is value-neutral;
    the single explicit boundary is the final nearest-even
    `ondsp.round_shift` by `16 + m`. The output reading is the
    unnormalized transform at `frac = 14 - m` (for example Q5.11 at
    `N = 8`). The exported value lies in `[-16384, 16383]` — the negative
    bound is reached exactly once, by the all-minimum input on the `k = 0`
    row through the `-16383.5` half-even tie — so the declared saturation
    is provably unreachable. Normalization conventions are the consumer's
    explicit follow-up quantization, not part of this contract.

    The executable profile pins `input_numeric` to signed Q1.15 and
    `output_numeric` to signed i16 with `frac = 14 - m` derived from the
    result extent.
  }];
  let arguments = (ins AnyRankedTensor:$input,
                   Ondsp_NumericAttr:$input_numeric,
                   Ondsp_NumericAttr:$output_numeric);
  let results = (outs AnyRankedTensor:$result);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def Ondrix_MovingAverageOp : Ondrix_Op<"moving_average", [Pure]> {
  let summary = "Valid-boundary moving average with one exact rounding";
  let description = [{
    `ondrix.moving_average` computes the valid-boundary sliding mean of a
    static rank-1 signed-Q1.15 tensor with a static power-of-two window
    `K = 2^m` in `[2, 64]`:

    ```text
    y[n] = round_shift(sum_{i=0}^{K-1} x[n + i], m),  n = 0..N-K
    ```

    The window sum is exact in i64 and the single explicit boundary is the
    final nearest-even `ondsp.round_shift` by `m`; a mean of Q1.15 values
    stays inside Q1.15, so the declared saturation is unreachable. This
    contract is deliberately NOT the equal-tap FIR
    `sum x[n+i] * q15(1/K)`: quantizing `1/K` first is a different
    program with a different result (a gain-merge-class divergence), which
    is why the operation exists instead of a rewrite onto `ondrix.fir`.
    Non-power-of-two windows require a rounded-division primitive and fail
    closed until one has a consumer.
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
  let summary = "Fixed-point sine over Q15 turn phases, table plus interpolation";
  let description = [{
    `ondrix.sine` maps a static rank-1 tensor of Q15 phases (`N` in
    `[1, 4096]`) elementwise to Q1.15 sines. A raw phase `p` is read as
    the unsigned 16-bit value `u = p mod 2^16` covering one full turn
    (`angle = 2*pi*u/65536`), so two's-complement wraparound IS phase
    wraparound and phase arithmetic stays exact. Per element:

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
  }];
}

def Ondrix_CosineOp : Ondrix_TrigOp<"cosine"> {
  let summary = "Fixed-point cosine over Q15 turn phases, table plus interpolation";
  let description = [{
    `ondrix.cosine` is `ondrix.sine` evaluated at `u + 16384 mod 2^16`
    (the exact quarter-turn phase advance): identical table, index and
    interpolation contract, and boundaries — see `ondrix.sine`. The two
    operations therefore agree bit-exactly under the phase identity
    `cos(u) = sin(u + quarter turn)`, which the exhaustive gate checks.
  }];
}

def Ondrix_GoertzelOp : Ondrix_Op<"goertzel", [Pure]> {
  let summary = "Single-bin Goertzel energy with quantized recursion state";
  let description = [{
    `ondrix.goertzel` computes the squared magnitude of one DFT bin `k`
    of a static rank-1 signed-Q1.15 tensor of length `N` in `[2, 4096]`
    with `0 <= k <= N/2`, through the quantized Goertzel recursion:

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
    observable state quantization; the final energy is the exact i64
    value of the integer expression (`|energy| < 3 * 2^30`), returned as
    `tensor<1xi64>` with no further rounding. The coefficient is
    admissible only under the `2^-20` quantization tie guard; other bins
    fail closed in lowering. Detecting one tone this way takes two words
    of recursion state instead of a full FFT's O(N) storage, at the cost
    of a different (also observable) quantization path — the Goertzel
    energy of a bin is NOT the squared magnitude of the corresponding
    staged-scaling FFT bin.
  }];
  let arguments = (ins AnyRankedTensor:$input,
                   I64Attr:$bin,
                   Ondsp_NumericAttr:$numeric,
                   Ondsp_RoundingModeAttr:$rounding);
  let results = (outs AnyRankedTensor:$energy);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def Ondrix_MatmulOp : Ondrix_Op<"matmul", [Pure]> {
  let summary = "Fixed-point matrix product with one boundary per element";
  let description = [{
    `ondrix.matmul` multiplies static signed-Q1.15 matrices
    `tensor<MxKxi16> x tensor<KxNxi16> -> tensor<MxNxi16>` with all three
    dimensions in `[1, 64]`, per output element:

    ```text
    C[i][j] = sat_i16(round_half_even(sum_k A[i][k] * B[k][j] / 2^15))
    ```

    Every full product and the K-term sum are exact in i64
    (`|sum| <= 64 * 2^30 = 2^36`), so the single observable boundary per
    element is the final nearest-even saturating requantization — the same
    contract shape as the FIR dot product, extended to a rank-2 result.
    Saturation is declared and reachable (for example a full row of
    `-32768` against a matching column). Reassociating the exact K-sum is
    value-neutral by construction; moving or splitting the requantization
    boundary is not.
  }];
  let arguments = (ins AnyRankedTensor:$lhs,
                   AnyRankedTensor:$rhs,
                   Ondsp_NumericAttr:$numeric,
                   Ondsp_RoundingModeAttr:$rounding);
  let results = (outs AnyRankedTensor:$result);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def Ondrix_RmsOp : Ondrix_Op<"rms", [Pure]> {
  let summary = "Fixed-point root-mean-square with two explicit boundaries";
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
  }];
  let arguments = (ins AnyRankedTensor:$input,
                   Ondsp_NumericAttr:$numeric,
                   Ondsp_RoundingModeAttr:$rounding);
  let results = (outs AnyRankedTensor:$result);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def Ondrix_GainOp : Ondrix_Op<"gain", [Pure]> {
  let summary = "Elementwise constant Q15 gain with one explicit requantization";
  let description = [{
    `ondrix.gain` scales a static rank-1 signed-Q1.15 tensor by one raw
    Q1.15 constant `gain` in `[-32768, 32767]`, with per-element semantics:

    ```text
    y[i] = saturate_i16(round(x[i] * gain / 2^15))
    ```

    The product is exact (`|x * gain| <= 2^30`) and the single explicit
    boundary is the requantization back to Q1.15; saturation is declared
    and reachable only for `x = gain = -32768`.

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
  }];
  let arguments = (ins AnyRankedTensor:$input,
                   I64Attr:$gain,
                   Ondsp_NumericAttr:$numeric,
                   Ondsp_RoundingModeAttr:$rounding);
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
  }];
  let arguments = (ins AnyRankedTensor:$input,
                   AnyRankedTensor:$desired,
                   AnyRankedTensor:$weights,
                   I64Attr:$step_size,
                   Ondsp_NumericAttr:$numeric,
                   Ondsp_RoundingModeAttr:$rounding);
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

    The executable profile is limited to static rank-1 `tensor<Nxi32>`
    packed input to `tensor<Nxi16>` output with `N` in `[1, 4096]`, the
    imag-high/real-low packed layout, signed Q15, and `toward_negative` or
    `nearest_even` rounding (the integer square root has no reachable
    nearest ties).
  }];
  let arguments = (ins AnyRankedTensor:$input,
                   Ondsp_CxLayoutAttr:$layout,
                   Ondsp_NumericAttr:$numeric,
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

    Each real coefficient is quantized once to signed Q1.15 with
    round-half-even and declared saturation at the +1.0 window center. The
    operation is a compile-time design intent with no runtime form: the
    opt-in `evaluate-ondrix-fir-design` pass replaces it with a constant
    coefficient tensor, and generation fails closed when any coefficient
    is closer than `2^-20` Q15 LSB to a rounding tie, so the emitted table
    provably equals the real-valued definition and is exactly symmetric.

    The result is a static rank-1 `tensor<Nxi16>` with `2 <= N <= 4096` and
    `numeric = #ondsp.fixed<signed, storage = i16, frac = 15>`.
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
    saturates to 32767 by declared convention. Quantization, the
    fail-closed `2^-20` tie guard, symmetry, and the
    `evaluate-ondrix-fir-design` consumer follow `ondrix.window_hamming`.

    The result is a static rank-1 `tensor<Nxi16>` with `2 <= N <= 4096` and
    `numeric = #ondsp.fixed<signed, storage = i16, frac = 15>`.
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
    the fail-closed `2^-20` tie guard, symmetry, and the
    `evaluate-ondrix-fir-design` consumer follow `ondrix.window_hamming`.

    The result is a static rank-1 `tensor<Nxi16>` with `2 <= N <= 4096` and
    `numeric = #ondsp.fixed<signed, storage = i16, frac = 15>`.
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
    binary64 error budget; the fail-closed `2^-20` quantization tie guard
    backstops the whole evaluation chain exactly as for the other window
    designs. The endpoints are `1 / I0(beta)` (nonzero) and the +1.0
    center of odd lengths saturates to 32767 by declared convention.

    The result is a static rank-1 `tensor<Nxi16>` with `2 <= N <= 4096`
    and `numeric = #ondsp.fixed<signed, storage = i16, frac = 15>`.
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
    Q1.15 with round-half-even and declared saturation; there is no
    quantize-then-combine step. The operation is a compile-time design
    intent evaluated by the opt-in `evaluate-ondrix-fir-design` pass, which
    fails closed when any coefficient is closer than `2^-20` Q15 LSB to a
    rounding tie. Under that guard the emitted integer table equals the
    real-valued definition and is exactly symmetric, so it feeds the
    existing constant-coefficient analysis, symmetric-pairing proofs, and
    proof-gated Vector paths unchanged.

    The result is a static rank-1 `tensor<Nxi16>` with odd
    `3 <= N <= 4095` and
    `numeric = #ondsp.fixed<signed, storage = i16, frac = 15>`.
  }];
  let arguments = (ins Ondrix_FirDesignResponseAttr:$response,
                   I64Attr:$cutoff_num, I64Attr:$cutoff_den,
                   Ondsp_NumericAttr:$numeric);
  let results = (outs AnyRankedTensor:$coefficients);
  let hasVerifier = 1;
  let assemblyFormat = "attr-dict `:` type($coefficients)";
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
