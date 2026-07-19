# Full-Output FIR Architecture Prototype

This note records the architecture experiment and the first experimental
full-output operation. It does not define a source-language binding.

## First Operation Contract

`ondrix.fir_filter` is the first contract-complete full-output seed. It uses
rank-1 tensor value semantics, implements `DestinationStyleOpInterface`, and
currently requires an explicit `#ondrix.fir_boundary<valid>` policy. For input
length `N` and coefficient length `K`, it requires `K > 0`, `N >= K`, and an
init/result extent of `N - K + 1`. Every output starts from numeric zero and
pairs `input[n + k]` with `coeffs[k]` in increasing `k` order. Init element
values are not part of the mathematical result.

The tensor-only boundary is deliberate. MLIR 17 destination style does not
provide a general proof that a destination memref is disjoint from input
memrefs. Tensor values therefore close the source-level alias contract without
an unverifiable no-alias promise. One-Shot Bufferize may lower the resulting
upstream tensor and SCF program after all Ondrix and Ondsp semantics have been
made explicit. When coefficients and the destination operand are the same SSA
tensor, bufferization must preserve the read-only coefficient value, inserting
a copy when needed. The bufferized ABI may still reuse the init operand's
backing buffer for the result. The executable regression suite covers this
aliasing case and checks results against the original coefficient values.

Fixed-point forms carry the input numeric policy, product selection,
accumulator type, destination numeric policy, rounding, and destination
overflow. Floating-point forms carry only their floating-point evaluation
policy and reject fixed-point lifecycle attributes. The generic lowering emits
an output-axis `scf.for`, an increasing-index tap `scf.for`, explicit Ondsp
accumulator updates and export for fixed point, and the selected FP update for
floating point.

## Prototype Contracts

The valid-mode prototype computes:

```text
output_length = input_length - coefficient_length + 1
output[n] = fir_sample(input[n : n + coefficient_length], coefficients)
```

It requires at least one coefficient, an input long enough to contain one
window, and an output buffer with exactly the required extent.

The full-boundary Q15 scalar spike uses zero padding and computes:

```text
output_length = input_length + coefficient_length - 1
input_index = output_index + tap - (coefficient_length - 1)
output[output_index] = ordered_sum(
  input[input_index] * coefficients[tap]
  for taps whose input_index is in bounds)
```

It requires nonempty input and coefficients. These choices are experimental;
in particular, empty-input behavior and coefficient indexing have not been
admitted as a stable source contract.

For both older memref prototype forms, output must not alias input or
coefficients; the two read-only operands may alias each other. This restriction
does not apply to tensor-value `ondrix.fir_filter`, whose bufferization must
preserve tensor SSA semantics as described above. Stride and dilation are one
at the algorithm level; scalar experiments also use non-unit physical memref
strides to validate descriptor handling.

The implementation intentionally composes existing IR:

```text
outer scf.for over output samples
  -> memref.subview for the current input window
  -> ondrix.fir for one ordered tap reduction
  -> explicit ondsp.acc_export
  -> memref.store
```

Tracked AOT tests cover valid-mode signed Q15 and Q31, ordered f32 FMA,
dynamic extents, nonzero descriptor offsets, unit-stride fixed-point Vector
chunking, dynamic-stride scalar fallback, accumulator overflow, and
process-level rejection of invalid dynamic extents. A separate Q15 execution
case covers both left and right zero-padded full boundaries.

## Architecture Result

The experiment establishes that the existing single-sample operation and
Ondsp tap-reduction semantics remain reusable inside a full-output kernel.
The new work is output-domain iteration, window construction, boundary
handling, destination effects, and aliasing. It is not a replacement for the
ordered inner reduction.

Valid windows directly reuse `ondrix.fir` through `memref.subview`. Padded
boundaries cannot be represented by a contiguous subview; the full-boundary
spike instead builds an ordered, bounds-guarded `ondsp.mac` chain. Both forms
consume the same numeric update semantics.

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

A full-output Ondrix operation can expose destination style and, later, tiling
over output axes. `ondrix.fir_filter` now establishes the destination-style
part. Its first lowering deliberately emits scalar tensor extracts instead of
calling the memref-only `ondsp.reduce_mac` consumer. A later buffer or tiled
lowering may form valid windows and reuse that reduction when it can preserve
the same ordered tap semantics. Linalg reduction lowering is legal only after
the concrete numeric policy proves that changing the update order is exact.

## Unresolved Stable Contract

The following remain outside the first valid tensor contract:

- equations and verification for same and full boundary modes;
- stride, dilation, and coefficient indexing direction;
- a buffer-semantics form with explicit alias and in-place legality;
- state ownership for streaming execution;
- output-axis tiling and fusion behavior.

Dynamic extent relationships are execution preconditions. The generic lowering
keeps all three checks in a dominating position before either loop and aborts
when an observed execution violates them. Because the source operation is pure,
these diagnostic checks need not survive when the operation and its result are
dead. Extent arithmetic is overflow-safe under the verified `K > 0` and
`N >= K` preconditions because `N - K + 1 <= N`.
Executable Q15, Q31, and ordered f32 tests pass through Ondrix lowering,
fixed-point finalization, One-Shot Bufferize, LLVM IR, object generation, C
linkage, and process execution. The older memref compositions remain useful as
architecture tests for tap-axis Vector reuse and padded boundaries.
