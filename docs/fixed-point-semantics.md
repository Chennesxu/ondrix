# Fixed-Point Semantics

This document defines the public fixed-point product and accumulator contract.
The executable scalar and fixed-width Vector paths implement signed-Q15 full
products and signed-Q31 full/raw-high products.

## Storage and Interpretation

Q15 values use signless `i16` MLIR storage. Signedness belongs to the Ondsp
numeric attribute rather than the builtin integer type:

```mlir
#ondsp.fixed<signed, storage = i16, frac = 15>
```

For raw two's-complement bits `x`, the represented value is:

```text
value(x) = signed(x) * 2^-15
```

## Product Selection

For two signed operands with storage width `W`, fractional position `F`, and
exact signed `2W`-bit product `P`, Ondsp defines:

```text
full.raw      = P
full.width    = 2W
full.frac     = 2F

high_raw.raw   = trunc_W(ashr(P, W))
high_raw.width = W
high_raw.frac  = 2F - W
```

`high_raw` is an arithmetic raw-bit selection. It does not imply doubling,
rounding, saturation, or conversion back to the operand fractional position.
For signed Q31 (`W = 32`, `F = 31`), `high_raw` therefore produces a Q30 term.
A fractional or doubled-high multiply requires a separate explicit policy and
is not currently part of the public contract.

The executable Q15 path selects `full`, producing a signed 32-bit raw value
with `frac = 30`. No rounding or saturation occurs during multiplication.

The executable Q31 paths support two separate domains:

- `full`: signed i64/frac62 product and accumulator, explicitly exported to
  signed i32/frac31;
- `high_raw`: signed i32/frac30 term accumulated in signed i40/frac30 state,
  explicitly exported as signed i32/frac30.

The raw-high path cannot be exported directly as Q31 because value-preserving
`acc_export` does not increase the accumulator fractional position.

## Accumulator Updates

The executable slice uses an explicit accumulator domain such as:

```mlir
!ondsp.acc<storage = i40, frac = 30, signed,
             update_overflow = saturate>
```

For accumulator raw value `a` and exact product `p`, `ondsp.mac` and
`ondsp.mac_sub` first compute an unbounded mathematical update:

```text
mac:     u = a + p
mac_sub: u = a - p
```

The accumulator's `update_overflow` policy is then applied at its declared
storage width:

```text
wrap(u, W)     = u modulo 2^W, interpreted as signed W-bit two's complement
saturate(u, W) = clamp(u, -2^(W-1), 2^(W-1)-1)
```

Update overflow is distinct from overflow when exporting the accumulator to a
destination storage type.

## Ordered Reduction

`ondsp.reduce_mac` consumes an explicit initial accumulator and two
equal-length rank-1 operands. Its universal semantics are an
increasing-index left fold:

```text
a[0]   = initial
a[i+1] = update(a[i], lhs[i] * rhs[i])
result = a[length]
```

The order is observable for saturating accumulators. A transformation may
reassociate the reduction only when it proves equivalence under the declared
numeric contract.

Wrapping updates form fixed-width modular addition, so horizontal reduction is
exact when product selection, extension, and truncation points are unchanged.
Saturating updates preserve lane order unless range analysis proves that every
source and transformed intermediate sum remains in range.

## Transform Equivalence

Ondsp records algebraic transform exactness (`exact` or `illegal`) separately
from its justification, such as an algebraic identity or fixed-width modulo
arithmetic. A range proof is owned by the analysis that derives it; it is not a
public Ondsp legality value, source attribute, or unchecked rewrite hint.
Bounded-error transforms remain unsupported until an explicit error-bound
object and composition rules are defined.

For a proven-no-overflow reassociation, the range analysis must establish that
every prefix interval in both the original update order and the proposed
update order fits the signed accumulator range. Merely proving that the final
mathematical sum fits is insufficient. For symmetric FIR pairing, the planner
accepts the complete raw coefficient sequence and derives the original and
candidate APInt intervals internally from the full signed input domain. It
returns a move-only plan bound to the operation, policies, accumulator, and
coefficient sequence rather than an independent proof token or reusable
boolean. Validated facts are exposed only to a one-shot consumer. The Ondsp
semantic classifier separately establishes the distributive identity.

Constant FIR specialization consumes this plan for signed full-product
symmetry. It derives each tap interval from the complete signed input-storage
range and the immutable raw coefficient, then validates both schedules before
rewriting. Saturating pairing remains disabled when any prefix may overflow,
for raw-high products, and while zero-tap elimination would need to be composed
with the pairing proof.

The constant-reduction chunk planner applies the same ownership rule to Vector
reassociation. It internally groups complete product intervals by the selected
fixed Vector width, retains the original scalar tail, and proves every prefix
of both schedules. A successful move-only plan authorizes horizontal chunk
sums only for the bound operation, policies, coefficients, and width.

### Constant FIR Pipeline Placement

Constant FIR specialization is an opt-in algorithm-layer branch, not a default
canonicalization. It consumes `ondrix.fir` and emits explicit loads plus
`ondsp.mac` or `ondsp.acc_add_term`, so running it also consumes the structured
reduction recognized by the generic Vector path. The current OrtumCore
conversion does not consume `ondsp.acc_add_term`; the supported consumer for a
specialized FIR is therefore the generic fixed-point scalar finalizer.

Until a cost model or target profile selects among alternatives, pipelines must
keep these choices explicit:

```text
generic CPU:  ondrix.fir -> ondsp.reduce_mac
              -> proven constant saturating chunks, or ordered Vector/scalar fallback
specialized:  ondrix.fir -> constant FIR specialization -> scalar lowering
target:       retain structured intent until a proven target capability matches
```

A default pipeline must not run constant specialization before generic Vector
or target selection merely because immutable coefficients are available.

## Import and Export

`ondsp.acc_import` is an exact value-preserving conversion into the
accumulator fractional domain. `ondsp.acc_export` is the only supported way
to leave that domain and explicitly declares:

- destination fixed-point format;
- rounding mode;
- destination overflow policy.

The implemented export rounding modes are `toward_negative`,
`toward_zero`, and `nearest_even`. Destination overflow is either `wrap`
or `saturate`.

## Vector Lowering

Unit-stride rank-1 memref reductions are split into fixed-width chunks and a
scalar tail.

- Saturating reductions with immutable coefficients may use horizontal chunks
  only when complete source and chunk-prefix analysis proves overflow is
  unreachable. Other saturating reductions vectorize loads and products, then
  apply lane updates in increasing order.
- Wrapping reductions may horizontally reduce widened products because Ondsp
  classifies the reassociation as exact modulo the accumulator width.
- Memrefs without a statically known unit minor stride remain on the scalar
  path.
- Q31 `high_raw` computes the exact signed i64 product before selecting the
  arithmetic high i32 half; no implicit fractional doubling is introduced.
- Generic LLVM Vector lowering currently supports only the default address
  space. Other memory spaces remain on the scalar path until a target-specific
  Vector mapping exists; invalid integer spaces fail before LLVM lowering.

Vector IR expresses semantic lane operations, not a performance guarantee.
Targets without native wide-lane multiplication may legally scalarize a Q31
vector product while preserving the same public numeric contract.

## Executable Evidence

The contract is checked by:

- the APInt reference implementation in
  [`lib/Support/FixedPointSemantics.cpp`](../lib/Support/FixedPointSemantics.cpp);
- scalar and Vector conversion tests under [`test/Conversion`](../test/Conversion);
- the Q15 OrtumCore capability emulator, which expands target operations back
  to this Ondsp contract before scalar finalization;
- PIC object and independent C-reference tests under
  [`test/Execution`](../test/Execution).

The execution corpus covers accumulator overflow, wrap and saturation,
order-sensitive reductions, dynamic lengths, Vector tails, nonzero seeds,
nonzero unit-stride offsets, scalar fallback, Q31 full-product rounding, and
signed Q31 raw-high bit selection.
