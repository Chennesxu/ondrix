# What Ondrix Claims, and How to Check It

This is the shortest complete path through the project. It states the one
claim the repository is built around, shows it on a running example, and
defines how strongly each statement in the rest of the documentation is
backed. Everything here links to code and tests in this repository; nothing
here requires trusting prose.

## The claim

**Fixed-point DSP semantics live at individual accumulator updates and
observable requantization boundaries — not in storage types, final results,
or a global rounding flag — and once those per-event, per-boundary semantics
are declared contracts, they become parameters of transformation legality:
the compiler can prove which rewrites preserve the declared finite-precision
program, apply exactly those, and refuse the rest.**

Three consequences follow, and each is mechanized here:

1. Optimizations that are identities over real arithmetic are *observably
   false* under the declared contracts, with pinned counterexamples.
2. Changing one contract value — the rounding mode of a boundary, the
   overflow behavior of an update — changes *which* rewrites are legal, not
   merely which bits come out.
3. A program compiled through different implementations of the same
   contracts (scalar, vector, abstract target capabilities) is bit-exact
   across them, so any divergence is a checkable defect, never an accepted
   tolerance.

### What this project does NOT claim

- Not a formally verified compiler: legality analyses are sufficient
  conditions checked by independent differential execution, not end-to-end
  machine-checked proofs.
- Not a complete DSP language: the frontend exposes a bounded set of
  algorithm contracts (see `status.md` for the exact slice).
- Not a claim of universal target performance; performance work is scoped
  to the schedules the legality framework admits.
- Not "the first fixed-point MLIR dialect": layered fixed-point IRs and
  rich fixed-point type systems exist. The specific position defended here
  is per-update/per-boundary contracts driving transformation legality.

## The running example

Four lines of `.ox`, every finite-precision choice spelled at the call site
(`test/Frontend/Inputs/q15_fir_filter_explicit_contract.ox`):

```python
def q15_fir_filter_explicit_contract(
    input: tensor[q15,64], coefficients: tensor[q15,8]) -> tensor[q15,57]:
  return fir_filter(input, coefficients, boundary=valid,
                    accumulator=exact[40,saturate], rounding=nearest_ties_positive, overflow=saturate)
```

**Layer 1 — `ondrix`: the algorithm and its contract, nothing else.** One
operation carries the whole declaration; order, boundary, and every numeric
knob are attributes, not compiler behavior:

```mlir
%1 = ondrix.fir_filter %input, %coeffs, %dest {
    accumulator = !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>,
    boundary = #ondrix.fir_boundary<valid>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    product = #ondsp.product<full>,
    rounding = #ondsp.rounding<nearest_ties_positive>,
    overflow = #ondsp.overflow<saturate>,
    dst = #ondsp.fixed<signed, storage = i16, frac = 15>}
```

**Layer 2 — `ondsp`: the numeric event graph.** The contract becomes
explicit events: one `ondsp.mac` per product (an i32-exact product added to
the i40 accumulator, *saturating on every update*), and one `ondsp.acc_export`
per output boundary (divide by 2^15 under the declared rounding, then the
declared destination overflow):

```mlir
%acc = scf.for %t = %c0 to %taps step %c1 iter_args(%a = %zero) -> (!ondsp.acc<...>) {
  %next = ondsp.mac %a, %sample, %coefficient {
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    product = #ondsp.product<full>} : (...)
  scf.yield %next
}
%out = ondsp.acc_export %acc {
  dst = #ondsp.fixed<signed, storage = i16, frac = 15>,
  rounding = #ondsp.rounding<nearest_ties_positive>,
  overflow = #ondsp.overflow<saturate>} : (...) -> i16
```

This layer is directly executable through the generic scalar lowering
(`--convert-ondsp-fixed-to-scalar`), which is the semantic authority every
other implementation is measured against.

**Layer 3 — `ortumcore`: proven target capabilities only.** The same kernel
maps onto an abstract DSP accumulator whose admitted behavior is exactly the
saturating i40 MAC and a shifted saturating readout. The target has no
native ties-positive export, so the conversion *composes* one from what is
proven — the floor readout at shift−1 plus one increment-and-halve — and
admits that composition only where its exactness argument holds (the
argument lives in the `ConvertOndspToOrtumCore` pass description):

```mlir
%acc  = scf.for ... {
  %next = ortumcore.mac_add %a, %sample, %coefficient : (...)
  scf.yield %next
}
%raw  = ortumcore.acc_out %acc {shift = 14} : (!ortumcore.acc) -> i32
%wide = arith.extsi %raw : i32 to i64
%inc  = arith.addi %wide, %c1
%half = arith.shrsi %inc, %c1        // floor((acc + 2^14) / 2^15), exactly
```

Anything the target has not proven fails closed: the conversion rejects the
program instead of approximating it.

## What the contracts buy: three rewrites

**1. A real-arithmetic identity, observably false.** Reordering a summation
is an identity over reals. Under per-update saturation it is not: an ordered
prefix can clip where a chunked order does not (or vice versa), and the
repository pins executable counterexamples rather than asserting the fact
(`test/Execution/legality_*_aot.mlir`, the finite-precision legality
counterexample gates). This is why "the final sum fits" is never accepted as
a reorder justification.

**2. A rewrite admitted by proof, not by optimism.** Saturating reductions
may still be vectorized when a subject-bound prefix-range analysis proves
that *every prefix of both the original and the candidate order* stays inside
the accumulator range; the plan is a move-only authorization bound to the
specific subject, coefficients, and chunk width, and is re-derived on the
original IR when replayed (`--vectorize-ondsp-constant-saturating-memref-reduce`,
`test/Transforms/widen_exact_accumulators.mlir` for the same zero-trust style
applied to accumulator retyping). The same machinery makes rounding a
*legality* parameter: the set of gain cascades that provably merge differs
in both directions between `nearest_even` and `nearest_ties_positive`, with
the divergence counts and first counterexamples pinned in
`test/Transforms/merge_gain_cascades.mlir`.

**3. A refusal you can read.** When the proof does not go through — a
dynamic coefficient, a reachable saturation, a multi-use intermediate — the
transform leaves the ordered program in place or the conversion rejects the
module with the failed obligation named. Fail-closed is the default failure
mode everywhere: frontend, conversions, transforms, target adapters.

## How claims are checked

Executable claims are gated by up to three independent legs plus recorded
evidence:

```
                 ondsp contract (the declared program)
                /                                     \
   generic scalar lowering                target-capability composition
   (semantic authority)                   (ortumcore + public emulation)
                \                                     /
                 == bit-exact ==      == bit-exact ==
                          \             /
                     independent C reference
              (separately written arithmetic, not a
               re-export of the compiler's helpers)
```

The scalar leg and the emulation leg share the generic lowering underneath,
so the C reference is the leg that rules out common-mode error: its
references use different formulations by design (floor *division* against
the compiler's shifts, digit-walk bit reversal against butterfly swaps,
add-half rounding against quotient/remainder classification). Tests are
*discriminating witnesses*: each named case is chosen so the rejected
semantics would produce a different answer (floor vs truncation, ties-even
vs ties-positive, saturate vs wrap, lane independence vs crosstalk), and
many gates carry self-checks proving the corpus still discriminates.
A private target backend maps the same `ortumcore` capabilities to real
instruction streams and is validated by the same differential method outside
this repository; nothing in the public claims depends on it.

### Evidence levels

Statements in `status.md` are backed at one of four levels. When reading any
capability row, identify its level; the levels are deliberately not
interchangeable.

| Level | Meaning | Example |
|---|---|---|
| **E1 exhaustive** | Every input of a finite domain executed and compared | 16-bit unary ops; 65536-phase sine/cosine tables |
| **E2 differential** | Discriminating witnesses + directed corpus across independent implementations | Q15 MAC/readout/export gates; FIR/FFT object+C gates |
| **E3 certified** | Static sufficient-condition proof, consumed fail-closed, with executable counterexamples for the refused side | prefix-range vectorization; gain-cascade merging; exact-accumulator widening |
| **E4 bounded-empirical** | Structured sampling with a stated error bound, domain too large to exhaust | `cx_phase` (2^32 input pairs; axes, diagonals, sweeps, pinned samples) |

E1 is proof by execution. E2 and E3 are strong evidence, not formal proof:
E2 depends on witness choice (which is why witnesses are named and argued),
E3 on the analysis being a genuinely sufficient condition (which is why every
certified transform also ships refusal witnesses). E4 is honest sampling and
is labeled as such where it appears.

## Where to go next

- `fixed-point-semantics.md` — the numeric contract definitions this
  document paraphrases.
- `frontend-language.md` — the `.ox` surface and its per-call-site policy
  parameters.
- `status.md` — the full capability inventory with per-row evidence; read
  it with the evidence levels above in hand.
- `include/ondrix/Conversion/Passes.td` and
  `include/ondrix/Transforms/Passes.td` — the soundness arguments for each
  conversion and certified transform, kept next to the code they justify.
