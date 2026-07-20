# Frontend Semantic Boundary

This document defines the intended boundary between the future `.ox` source
language, project dialects, and upstream MLIR dialects. It is a design
contract, not a statement that the source parser is implemented.

## Design Rule

The frontend must not mirror every source construct into a custom operation.
A custom project IR construct is justified only when it preserves algorithm,
numeric, or target-capability information that upstream MLIR cannot recover
after lowering. Within that split, Ondrix owns algorithm intent and Ondsp owns
numeric contracts.

Generic source constructs lower directly to upstream dialects:

| Source concept | MLIR ownership |
| --- | --- |
| Functions, calls, and returns | `func` |
| Structured control flow | `scf` and `cf` |
| Ordinary integer and floating-point arithmetic | `arith` and `math` |
| Buffers, views, and subviews | `memref` or `tensor` |
| Semantic lane vectors | builtin `vector` types and the Vector dialect |
| DSP algorithm intent | Ondrix dialect |
| Fixed-point products, accumulators, and export policy | Ondsp dialect |
| Proven target capabilities | OrtumCore dialect |

Physical registers, register tuples, encodings, scheduling, and instruction
selection remain private-backend concerns.

## Source-Level Algorithm Constructs

The stable source surface should remain smaller than the internal compiler
surface. Candidate source constructs include FIR, convolution/correlation,
FFT-family transforms, recursive filters, and resampling. Each construct must
define its output domain, indexing, state, and numeric boundaries before it is
bound to a stable Ondrix operation.

The current operations have different expected roles:

- `ondrix.fir` is an algorithm operation for one pre-windowed output sample.
- `ondrix.dot` may be frontend convenience syntax. Without axis, conjugation,
  batching, masking, or other retained intent, it may lower immediately to
  `ondsp.reduce_mac`.
- `ondrix.butterfly` is expected to be produced by transform decomposition;
  it need not be a user-facing source primitive.
- fixed-point casts and quantization are source expressions whose normalized
  semantics belong to Ondsp rather than the algorithm dialect.

Full-output and streaming FIR use distinct contracts for boundaries, output
shape, state, and aliasing. The experimental `ondrix.fir_filter` and
`ondrix.fir_stream` operations encode those separate contracts; a future
frontend must not infer either one from the single-sample operation.

## Source Types and Numeric Policy

The source language should provide an explicit fixed-point value type, with
convenience aliases such as Q15 and Q31. The unambiguous form records:

```text
fixed<signedness, storage-bits, fractional-bits>
```

Source syntax should normally avoid exposing normalized MLIR types such as
`!ondsp.acc`. High-level kernels instead choose one of two accumulator modes:

- `auto`: overflow is not part of the source behavior. The compiler must infer
  a sufficient width or reject the program when it cannot prove one.
- `exact`: width and per-update overflow are observable program semantics,
  suitable for bit-exact compatibility with an existing DSP kernel.

The normalized Ondsp IR always materializes the selected width, fractional
position, signedness, and update overflow. A target must not silently replace
`auto` exact accumulation with a narrower saturating capability.

Leaving an accumulator requires explicit destination format, rounding, and
overflow policy. Source defaults, if introduced, must be language defaults and
must not depend on the selected target.

## Compile-Time Algorithm Facts

The frontend should communicate facts through ordinary source semantics:

- immutable coefficients use a `const` declaration;
- windows and complete signals use distinct checked types or expressions;
- static extents remain compile-time shape information;
- stateful kernels expose their state and update order.

Properties such as zero taps, sparsity, symmetry, antisymmetry, repeated
coefficients, powers of two, and trivial twiddles are inferred from immutable
values. A user-authored promise must not enable a rewrite unless it is verified
or represented by an explicit checked assumption.

## Optimization Contract

The intended pipeline is:

```text
.ox parser and AST
  -> source type, shape, and numeric checking
  -> Ondrix algorithm intent plus upstream MLIR
  -> algorithm fact inference and transforms
  -> normalized Ondsp numeric semantics
  -> scalar, Vector, or OrtumCore consumer
```

An algorithm transform first identifies a mathematical identity. Ondsp then
classifies that identity under the concrete product, accumulator, rounding,
and overflow contract. Transformations must distinguish:

- bit-exact equivalence;
- exact fixed-width modular equivalence;
- equivalence requiring a no-overflow proof;
- bounded-error transforms only after an explicit error-bound contract exists;
- unsupported or illegal transformations.

Proof facts are compiler-produced analysis results, not unchecked rewrite
attributes. Optimized variants should normally be represented with the same
algorithm operations, normalized Ondsp operations, and upstream dialects,
rather than one custom operation per theorem or fusion pattern.

## Operation Admission Checklist

A new stable Ondrix operation requires all of the following:

1. A defined source-level construct or compiler-generated algorithm role.
2. Information that cannot be recovered from upstream MLIR after lowering.
3. Complete shape, state, memory, and numeric verifier rules.
4. A public generic lowering or executable consumer.
5. At least one real analysis, transform, or target consumer.

Operations that do not satisfy these conditions remain experimental or are
lowered directly to Ondsp/upstream MLIR.
