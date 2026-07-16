# Fixed-Point Semantics

This document defines the public fixed-point product and accumulator contract.
The executable scalar and fixed-width Vector paths currently implement its
signed-Q15 full-product subset.

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

- Saturating reductions vectorize loads and products, then apply lane updates
  in increasing order.
- Wrapping reductions may horizontally reduce widened products because Ondsp
  classifies the reassociation as exact modulo the accumulator width.
- Memrefs without a statically known unit minor stride remain on the scalar
  path.
- Default and representable nonnegative integer LLVM address spaces are
  supported. Target-specific memory-space attributes require a target mapping,
  and invalid integer spaces fail before LLVM lowering.

## Executable Evidence

The contract is checked by:

- the APInt reference implementation in
  [`lib/Support/FixedPointSemantics.cpp`](../lib/Support/FixedPointSemantics.cpp);
- scalar and Vector conversion tests under [`test/Conversion`](../test/Conversion);
- PIC object and independent C-reference tests under
  [`test/Execution`](../test/Execution).

The execution corpus covers accumulator overflow, wrap and saturation,
order-sensitive reductions, dynamic lengths, Vector tails, nonzero seeds,
nonzero unit-stride offsets, and scalar fallback.
