# Full-Output FIR Contract

`ondrix.fir_filter` is the experimental full-output FIR operation. It uses
rank-1 tensor value semantics, implements `DestinationStyleOpInterface`, and
supports `valid` and `full` boundary policies. Every output starts from
numeric zero and visits coefficients in increasing tap order.

## Output Equations

For an input of length `N` and a nonempty coefficient sequence of length `K`,
valid mode requires `N >= K` and computes:

```text
output_length = N - K + 1
output[n] = ordered_sum(
  input[n + k] * coefficients[k]
  for k = 0 .. K - 1)
```

Full mode requires a nonempty input. Without `output_origin`, it computes the
complete output:

```text
output_length = N + K - 1
global_output_index = n
input_index = global_output_index + k - (K - 1)
output[n] = ordered_sum(
  input[input_index] * coefficients[k]
  for k = 0 .. K - 1 where input_index is in bounds)
```

A compiler-generated full tile may instead carry `output_origin` and produce a
contiguous subrange. Its equation uses
`global_output_index = output_origin + n`; the tile verifier requires the
origin and local extent to remain within the complete output domain.

An out-of-range tap performs no update. This is equivalent to fixed-point zero
padding while avoiding an observable `0 * Inf` or `0 * NaN` operation for
floating point. Stride and dilation are one.

Fixed-point forms explicitly carry source numeric, product, accumulator,
destination numeric, rounding, and destination-overflow policies.
Floating-point forms carry an FP evaluation policy and reject fixed-point
lifecycle attributes.

## Tiling And Bufferization

The operation exposes only its parallel output axis through
`TilingInterface`. A valid tile reads the corresponding input halo. A full
tile retains the complete input and carries a compiler-generated global
`output_origin`. Neither form tiles or reassociates the ordered tap reduction.

The external bufferization model writes directly to the destination buffer.
Valid windows and the fully overlapping interior of full mode become
rank-1 `memref.subview` values consumed by `ondsp.reduce_mac`; padded full
edges retain guarded scalar updates. Unit-stride Q15/Q31 reductions may then
reuse existing ordered or proof-authorized Vector consumers. Non-unit physical
strides and f32 retain scalar fallback.

Tensor SSA semantics define the aliasing contract. Bufferization must preserve
read-only input and coefficient values when a destination aliases their
storage, using an out-of-place result or a dominating snapshot as required.
The current bufferized descriptor convention is an execution-test mechanism,
not a stable source or C ABI.

Dynamic extent relationships are execution preconditions. Lowerings may emit
diagnostic guards, but those guards are not observable algorithm behavior.

## Deliberate Limits

The current contract does not define:

- `same` boundary mode, stride, or dilation;
- a public buffer-semantics operation or stable ownership ABI;
- stateful streaming ownership;
- target-aware tile selection;
- general fusion guarantees.

The scalar-result `ondsp.reduce_mac` is not output-tileable: tap partitions
carry serial accumulator state unless reassociation is independently proven.
Output tiling therefore remains an Ondrix algorithm-layer responsibility,
while tap-axis legality remains an Ondsp numeric-layer responsibility.
