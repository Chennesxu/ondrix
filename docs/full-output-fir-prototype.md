# Full-Output FIR Architecture Prototype

This note records the architecture experiment and the first experimental
full-output operation. The `.ox` frontend now provides an experimental
tensor-value binding for valid output and statically shaped full output;
dynamic full output and a stable source or C ABI remain undefined.

## First Operation Contract

`ondrix.fir_filter` is the first contract-complete full-output seed. It uses
rank-1 tensor value semantics, implements `DestinationStyleOpInterface`, and
supports explicit `valid` and `full` boundary policies. Every output starts
from numeric zero and visits coefficients in increasing `k` order. Init
element values are not part of the mathematical result.

The tensor-only boundary is deliberate. MLIR 17 destination style does not
provide a general proof that a destination memref is disjoint from input
memrefs. Tensor values therefore close the source-level alias contract without
an unverifiable no-alias promise. One-Shot Bufferize may lower the resulting
upstream tensor and SCF program after all Ondrix and Ondsp semantics have been
made explicit. When coefficients and the destination operand are the same SSA
tensor, bufferization must preserve the read-only coefficient value by copying
the read operand or selecting an out-of-place result buffer. The bufferized ABI
may reuse the init operand's backing buffer when analysis proves that legal.
The executable regression suite covers both coefficient/init and input/init
aliasing and checks results against the original read-only values.

Fixed-point forms carry the input numeric policy, product selection,
accumulator type, destination numeric policy, rounding, and destination
overflow. Floating-point forms carry only their floating-point evaluation
policy and reject fixed-point lifecycle attributes. The generic tensor lowering
emits an output-axis `scf.for`, an increasing-index tap `scf.for`, explicit
Ondsp accumulator updates and export for fixed point, and the selected FP
update for floating point.

An external `BufferizableOpInterface` model provides the optimized buffer path.
It writes samples directly and represents each fully overlapping input window
and the coefficient sequence as rank-1 `memref.subview` values consumed by
`ondsp.reduce_mac`. Valid output and the interior of full output therefore
reuse the existing fixed-width Q15/Q31 Vector tap-reduction passes. Padded full
edges remain increasing-index guarded scalar updates. F32 keeps the ordered
scalar reduction until an exact FP Vector policy is implemented.

`ondrix.fir_filter` implements `TilingInterface` for its single parallel output
axis. A valid-boundary tile at `[offset, offset + size)` reads the input halo
`[offset, offset + size + K - 1)`. A full-boundary tile retains the complete
input and carries `offset` as its explicit global `output_origin`. Both retain
the complete coefficient tensor and use the corresponding init slice. The
ordered tap reduction remains inside each tiled operation. The opt-in
`tile-ondrix-fir-filter` pass delegates SCF loop construction, tail sizes, and
destination insertion to MLIR's interface tiler; it does not tile or
reassociate the tap axis.

## Prototype Contracts

The valid-mode prototype computes:

```text
output_length = input_length - coefficient_length + 1
output[n] = fir_sample(input[n : n + coefficient_length], coefficients)
```

It requires at least one coefficient, an input long enough to contain one
window, and an output buffer with exactly the required extent.

Full boundary uses zero padding and computes:

```text
output_length = input_length + coefficient_length - 1
input_index = output_index + tap - (coefficient_length - 1)
output[output_index] = ordered_sum(
  input[input_index] * coefficients[tap]
  for taps whose input_index is in bounds)
```

It requires nonempty input and coefficients. An out-of-range tap performs no
accumulator update. This is equivalent to zero padding for fixed point, while
avoiding evaluation of `0 * Inf` or `0 * NaN` for floating point. This indexing
convention is part of the experimental Ondrix contract and makes outputs
`K - 1` through `N - 1` identical to the valid-boundary equation when `N >= K`.

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

Tracked AOT tests cover valid and full signed Q15/Q31, ordered f32 FMA,
dynamic extents, nonzero descriptor offsets, unit-stride fixed-point Vector
chunking, dynamic-stride scalar fallback, accumulator overflow, and
process-level rejection of invalid dynamic extents. Full-boundary execution
also covers `K > N`, both padded edges, Vectorized interiors, and tensor alias
conflicts. Floating-point padded edges include infinity, NaN, and signed-zero
cases that distinguish skipped updates from eagerly multiplying explicit zero
padding. The same full-boundary corpus executes untiled optimized, generic
tensor-to-scalar, and output-tiled optimized object pipelines.

## Architecture Result

The experiment establishes that the existing single-sample operation and
Ondsp tap-reduction semantics remain reusable inside a full-output kernel.
The new work is output-domain iteration, window construction, boundary
handling, destination effects, and aliasing. It is not a replacement for the
ordered inner reduction.

Valid windows directly reuse `ondrix.fir` or `ondsp.reduce_mac` through
`memref.subview`. Padded boundaries cannot be represented by one contiguous
subview. Full bufferization therefore partitions the output into a guarded
left edge, a fully overlapping `reduce_mac` interior, and a guarded right edge.
If `K > N`, the interior is empty and the two edge ranges still cover every
output exactly once. All forms consume the same numeric update semantics.

The output-tiling pass emits the three complete-output dynamic shape checks
once before the outer tile loop. Each public full-boundary tile also validates
that its explicit output origin and local extent lie within that complete
output before using the origin for local-to-global index mapping. Bufferization
derives each window and coefficient subview from the same coefficient-length
SSA value. A future internal witness may let compiler-generated tiles reuse
the dominating proof without repeating the tile-containment check.

The current scalar-result `ondsp.reduce_mac` is not a suitable destination-
style tiled operation: a tap tile has no independently insertable result, and
saturating updates generally require serial accumulator flow between tap
tiles. Tap chunking therefore remains a numeric transform with explicit
ordering or reassociation proof.

A full-output Ondrix operation can expose destination style over every boundary
mode and tiling where the tile contract is complete. `ondrix.fir_filter`
establishes destination style and output-only tiling for valid and full. Full
tiles carry the global output origin required to retain the padding equation.
Its external bufferization model forms fully overlapping windows and reuses the
memref-only `ondsp.reduce_mac` consumer while preserving the same ordered tap
semantics.
Linalg reduction lowering remains legal only after the concrete numeric policy
proves that changing the update order is exact.

MLIR's SCF tiler still materializes tensor destination insertion, but One-Shot
Bufferize maps the tiled source and destination to equivalent subviews. CSE and
canonicalization remove that self-copy. When coefficients alias the destination
tensor, bufferization instead creates one protective coefficient snapshot
before the outer tile loop; no allocation or copy remains inside the tile loop.

The separate `lower-rank-one-memref-copy-to-scf` pass remains a standalone AOT
fallback for unrelated rank-1 copies. It snapshots logical elements before
writing the destination, preserving offset, stride, and overlapping-view
semantics without a runtime `memrefCopy` dependency. It is not part of the
optimized FIR pipeline and is not a performance lowering.

## Unresolved Stable Contract

The following remain outside the first tensor contract:

- equations and verification for same boundary mode;
- stride and dilation;
- a public buffer-semantics form with explicit alias and in-place legality;
- state ownership for streaming execution;
- fusion behavior and target-aware output tile selection.

Dynamic extent relationships are execution preconditions. Generic lowering and
full-boundary bufferization validate them before computing output samples and
abort when an observed execution violates them. Output tiling establishes the
complete-output checks once before the outer tile loop; public tile operations
independently validate their subrange.
Because the source operation is pure, diagnostic checks need not survive when
the operation and its result are dead. Extent arithmetic is overflow-safe under
the verified `K > 0` and `N >= K` valid preconditions because `N - K + 1 <= N`.
Full shape guards recover `N` as `output_length - (K - 1)`, avoiding an
overflowing `N + K` intermediate. Guarded full indexing similarly uses the
first valid tap and a modular base rather than comparing an overflow-prone
`output + tap` sum. Executable untiled and output-tiled Q15, Q31, and ordered f32 tests pass through
Ondrix lowering or external bufferization, fixed-point finalization, LLVM IR,
object generation, C linkage, and process execution. The optimized tiled Q15
and Q31 path additionally executes existing fixed-width Vector tap chunking;
the alias regression verifies that any protective snapshot dominates the tile
loop. Full-boundary AOT separately executes Vector interiors, scalar edges,
short inputs, and alias-preserving out-of-place bufferization. The older memref
compositions remain useful as architecture tests for physical-stride fallback.
