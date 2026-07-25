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
    Computes one static four- or eight-point radix-2 decimation-in-time CFFT
    over packed signed-Q15 complex values. The input and result use natural
    index order and matching `tensor<Nxi32>` storage. Each packed element uses
    the layout and finite-precision butterfly contract of
    `ondrix.butterfly`.

    The transform recursively decomposes even and odd input indices, then
    combines the two half transforms in increasing frequency order:

      `(y[k], y[k + N/2]) = butterfly(even[k], odd[k], W_N^k)`

    Forward direction uses `W_N^k = exp(-2*pi*j*k/N)`, frozen to signed-Q15
    packed constants. Four points use `[1, -j]`. Eight points use
    `[1, 23170-j23170, -j, -23170-j23170]`.

    Inverse direction uses `conj(W_N^k)`. Four points use `[1, +j]`. Eight
    points use `[1, 23170+j23170, +j, -23170+j23170]`. Q15 `1` and `+j`
    use positive raw value 32767, while `-j` uses raw value -32768.

    Each butterfly applies the declared product and output requantization
    independently. A size-N transform therefore has `log2(N)` observable
    scaling boundaries; they must not be collapsed into one final
    real-arithmetic normalization. With the current one-bit output scale at
    every stage, inverse direction includes the conventional `1/N`
    normalization.

    The executable profile is deliberately limited to four or eight points,
    explicit forward or inverse direction, the imag-high/real-low packed
    layout, signed Q15, exact full products, and the same two scale policies
    as the executable butterfly. Dynamic planning, other sizes and numeric
    profiles require separate extensions.
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

def Ondrix_QuantizeOp : Ondrix_Op<"quantize", [Pure]> {
  let summary = "High-level numeric conversion intent";
  let arguments = (ins AnyType:$input, Ondsp_NumericAttr:$src,
                   Ondsp_NumericAttr:$dst);
  let results = (outs AnyType:$result);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

#endif
