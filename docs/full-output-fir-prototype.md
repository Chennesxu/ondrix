# Full-Output FIR Architecture Prototype

This note records an executable architecture experiment. It does not define a
stable Ondrix operation or source-language binding.

## Prototype Contract

The prototype computes a rank-1, valid-mode FIR output:

```text
output_length = input_length - coefficient_length + 1
output[n] = fir_sample(input[n : n + coefficient_length], coefficients)
```

It requires at least one coefficient, an input long enough to contain one
window, and an output buffer with exactly the required extent. Output must not
alias input or coefficients; the two read-only operands may alias each other.
Stride and dilation are one at the algorithm level; the scalar experiment also
uses non-unit physical memref strides to validate descriptor handling.

The implementation intentionally composes existing IR:

```text
outer scf.for over output samples
  -> memref.subview for the current input window
  -> ondrix.fir for one ordered tap reduction
  -> explicit ondsp.acc_export
  -> memref.store
```

The tracked AOT test covers signed Q15, dynamic extents, nonzero descriptor
offsets, unit-stride Vector chunking, dynamic-stride scalar fallback,
accumulator overflow, and process-level rejection of invalid dynamic extents.

## Architecture Result

The experiment establishes that the existing single-sample operation and
Ondsp tap-reduction semantics remain reusable inside a full-output kernel.
The new work is output-domain iteration, window construction, boundary
handling, destination effects, and aliasing. It is not a replacement for the
ordered inner reduction.

The current composition also repeats the dynamic window/coefficient length
check inside each output iteration. A stable full-output lowering should
establish that relationship once through its verifier, a runtime shape
witness, or an equivalent dominating check, then avoid rechecking it in every
single-sample reduction.

The current scalar-result `ondsp.reduce_mac` is not a suitable destination-
style tiled operation: a tap tile has no independently insertable result, and
saturating updates generally require serial accumulator flow between tap
tiles. Tap chunking therefore remains a numeric transform with explicit
ordering or reassociation proof.

A future full-output Ondrix operation may implement destination-style and
tiling interfaces over output axes. Its generic lowering should form outer
SCF loops and reuse `ondsp.reduce_mac` for each output window. Linalg reduction
lowering is legal only after the concrete numeric policy proves that changing
the update order is exact.

## Unresolved Stable Contract

The following must be resolved before admitting a stable full-output op:

- valid, same, and full boundary equations;
- stride, dilation, and coefficient indexing direction;
- destination tensor and memref forms;
- input/output aliasing and in-place legality;
- dynamic shape verification and runtime witnesses;
- state ownership for streaming execution;
- Q15, Q31, and floating-point policy coverage;
- output-axis tiling and fusion behavior.

Until then, the executable prototype remains test-only composition built from
upstream dialects and the existing single-sample FIR contract.
