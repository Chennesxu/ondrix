# f32 contract evidence ledger

What each `fast` declaration actually buys across the thirteen real,
non-transform operations of the f32 profile, and where a declaration is
admitted with no executed evidence behind it. Organised in three layers,
because the earlier one-row-per-operation table could not express the answer:
what a permission costs depends on the shape and target a site reaches, not on
the operation name.

## Facets

A declaration is not a schedule. Six things are distinct:

- **Base graph** — the event graph the operation's description states
  literally, in declared order: one rounded product per term, summed in order.
- **Legal set** — the graphs derivable from the base graph using only the
  rewrites the declaration permits. `off` and `fma` permit none, so their legal
  set is a single graph and they are exact contracts: bitwise reproducible for
  non-NaN outputs, NaN class preserved, signed zeros and infinities bitwise.
- **Selected** — the one member the lowering builds. A compiler policy, not a
  language rule; nothing forbids a different member.
- **Used** — which permissions were spent getting from the base graph to the
  selected graph. Selecting a fused event spends **F** even though the emitted
  operation carries no flag; rebuilding the tree across lanes spends **R**.
- **Emitted** — permissions left on the operations that reach the audit point.
  Spending and emitting are opposites: an emitted flag hands the choice to
  LLVM instead of making it. Emitted is `{}` everywhere, gated by
  `test/Permissions/fp_contract_permission_audit.mlir` on the translated IR.
- **Executed gate** — object-level evidence. An exact contract is pinned
  bitwise against an independent reference. A relaxed result is not pinned; its
  evidence is term conservation on a sub-domain where every derivable rebuild
  agrees.

The audit point is the translated `.ll`: this flow runs no LLVM middle end, so
permissions are final there. It is *not* the final point for the realized event
graph, because `llc` still consumes fast-math flags — a `reassoc` on `llvm.fma`
is enough for the X86 backend without +fma to de-fuse it, while AArch64 and the
32-bit ARM DSP targets keep it fused even with no fused instruction to keep
(`test/Target/fp_permission_fmf_*.ll`). Which graph runs is a per-backend
expansion policy that a delegated permission cannot bound, which is why emitted
is empty rather than merely bounded.

## The two permissions

`fast` permits **R** (`RebuildReductionTree`) and **F** (`FuseMultiplyAdd`).
This section is the normative statement; the dialect and pass descriptions
summarize it and do not extend it.

### R is relative to a designated reduction instance

R is not defined over syntactically additive expressions. It is defined over a
**designated reduction instance** `p` — an operation-defined semantic reduction
instance, named by operation semantics rather than by syntax, and parameterized
by the operation, its loop family, or an output index. "Instance" is not an
MLIR `Region`: a `dct` row and a `moving_average` window are instances without
being region objects. At the `ondsp` level one `ondsp.reduce_mac` operation is
one such instance; at the `ondrix` level it is the sum the operation's description states
literally, such as a `dct` row, a `moving_average` window, or the tap sum of
`lms`. An expression is not a reduction merely by containing additions.

Each instance carries an optional distinguished seed `s(p)` — the `initial`
operand where the description has one — an index domain `D(p)`, and one
**indexed term occurrence** `t(p, i)` per `i` in `D(p)`, with fixed operands and
a fixed index relation.

R permits replacing the additive tree of `p` with any binary addition tree whose
leaf sequence is

```text
[ s(p), t(p, pi(0)), ..., t(p, pi(N-1)) ]
```

for a permutation `pi` of `D(p)`, parenthesized arbitrarily. The seed occurs
exactly once and stays the first leaf; R does not permit moving it among the
terms, and no current mechanism needs that. Where the description states no
seed, the permutation ranges over all leaves.

An instance with a seed and an empty domain evaluates to the seed. An instance
with no seed must have a non-empty domain: a seedless empty reduction has no
declared result and is not admitted. No operation in the catalogue reaches that
corner — the seedless reductions are `dct` rows at a static power-of-two extent
in `[4, 64]`, `rms` at `N >= 2`, and `moving_average` at `K >= 2` — so this
closes the definition rather than a reachable case.

Permutation is the part associativity does not give. Lane `i` pairs term `i`
with term `i + W`, and reparenthesization alone preserves leaf order, so
associativity by itself would not authorize the schedule this compiler builds.
The difference is observable: over `[1e8, 1.0, 0, …, -1e8, …]` at N=16, W=8 the
source leaf order gives 0 and the lane partition gives 1.

Because the leaves are a bijection onto the term occurrences of `p`, an identity
leaf cannot be added. That is a consequence of the definition rather than a
second condition, so it is stated once here.

### R is anchored to the declared base graph, not to the current IR

The source of a rebuild is always the base graph the call site declared. A pass
may rebuild a tree another pass already rebuilt, but the leaves keep their
original `(p, i)` identities and admissibility is still settled against the same
base graph. Treating an intermediate as a new source would let a chain of
individually plausible steps leave the legal set of the declaration that
authorized them.

Today the anchor is structural rather than carried: the one R-consuming
transform matches `ondsp.reduce_mac` itself, so the designated instance is the
operation in front of it and there is no chain to lose. That is why no
provenance attribute exists yet. A second R consumer, or one that ran after the
tree was already rebuilt, would have to establish the original `(p, i)`
identities or refuse — this paragraph is the obligation it inherits, not a
mechanism the compiler already runs.

R does not permit mixing two reduction instances — two outputs, two samples, two
operations — splitting one instance into independent results, crossing a
non-additive boundary such as a division, a saturation, or an export, or
changing a term's operands or index relation.

A membership check may still compare graphs up to identities proved bitwise —
same non-NaN output for every input: multiplicand exchange in a rounded or
fused product, `x * 1.0 = x`, and `fma(x, 1.0, c) = c + x`. Two graphs equal
under that quotient are the same function, so the quotient admits no member
the definitions exclude; it does not extend to addition operands, whose order
carries the tree shape R governs.

### R and F compose over term occurrences, not IR nodes

**F** selects a fused multiply-add event for a term in place of a rounded
product followed by an addition. Under `{R, F}` the product of `t(p, i)` is
therefore no longer a separate operation in the result.

The bijection constrains term occurrences and their provenance, not whether a
distinct multiply survives in the emitted IR. F may fuse a term with the
addition that consumes it, but it may not delete, duplicate, or re-source that
term. In either order of application, every original term occurrence appears
exactly once in the selected graph.

## Layer 1 — language: what each operation's `fast` admits

| Operation | Legal set under `fast` |
| --- | --- |
| `gain` | one graph. A lone product has no addend to fuse and is not a designated reduction — the only operation whose three declarations denote the same events. |
| `moving_average` | the window sum is a designated reduction, so **R** applies for `K >= 2`; no product, so **F** never does. At `K = 2` R already admits distinct event graphs, but operand exchange in a two-operand IEEE addition is bitwise commutative for every non-NaN result. Any remaining difference is confined to the NaN result's representation or which NaN operand it is taken from — payload and sign — neither of which the contract specifies. Signalling-NaN behaviour is outside the contract, and is not claimed here to be order-distinguishing: the invalid condition is symmetric in the operands. The first difference in value appears at `K = 3`. |
| `dot`, `fir`, `fir_filter`, `conv1d`, `matmul`, `rms`, `dct`, `fir_decimate`, `fir_interpolate`, `lms` | products plus a designated reduction: both **R** and **F** apply. |
| `goertzel` | exactly two graphs. No part of the recursion is a designated reduction — the sample chain is a data dependence and the closing energy is a fixed expression, not a sum over an index domain — so only **F** applies, at one multiply-add site. |
| `sos_filter_tdf2` | larger than one graph — the biquad body has fusable multiply-adds — and no recurrence-level realization gate exists, so `fast` is refused at the verifier. |

Admission follows this layer, not the transform list: a singleton legal set is
admitted as provably inert, a larger one is admitted when a realization gate
exists for what is selected, and otherwise it fails closed.

## Layer 2 — mechanism: what each schedule spends

| Mechanism | Selected graph | Used |
| --- | --- | --- |
| ordered scalar | base graph, declared order | {} |
| scalar fused chain | one fused event per term, declared order | {F} |
| horizontal, separate terms | lane partition, rounded products | {R} |
| horizontal, fused terms | lane partition, fused lane accumulation | {R, F} |

Term selection inside the horizontal mechanism is read from the declared
`supports-vector-fma` capability, never from the host. The lane seed stays a
raw product either way, because a seed has no addend to fuse — selection is per
term and uniformity across a reduction is not a rule of the language.

## Layer 3 — reachability: which mechanism a site gets

At the default width, `supports-vector-fma=false`:

| Route | Mechanism | Used | Executed gate |
| --- | --- | --- | --- |
| `dot`, `fir`, `fir_filter`, `rms`, `fir_decimate` | horizontal, separate | {R} | mechanism only (`fp_fast_reduce_aot`) |
| `matmul` | column tile; the fused body is refused without the vector fused multiply-add, so this configuration keeps the scalar fused chain. Under `supports-vector-fma=true` the tile batches, and its unrolled inner axis rebuilds into interleaved chains, adding {R} | {F} | `fs_f32_solver` route audit, `matmul_f32.mlir` |
| `conv1d` correlation, K ≥ W | horizontal, separate | {R} | `f32_conv1d_aot`, term conservation |
| `conv1d` convolution | batched over the reversed taps; at `K >= 4` the declared-fast fold rebuilds into interleaved chains, adding {R}; below four taps each lane keeps its left fold, and outputs too few to batch keep the scalar fused chain | {F} or {F, R} by `K` | `fs_f32_solver` route audit (K=3 and K=8), `fp_filter_output_batching.mlir`, `f32_conv1d_aot` |
| `dct` | 128-bit: batched output tile, scalar fused rows, {F}; 256-bit: per-row horizontal rebuild over the transposed sibling table, adding {R} per row | {F} or {F, R} by width | `fs_f32_solver` route audit (N=8 and N=16 w256), `dct_f32.mlir` |
| `lms`, `fir_interpolate` | scalar fused | {F} | selection pinned |
| `goertzel` | scalar fused | {F} | `f32_goertzel_aot`, bitwise against `fma` plus a `.ll` permission pin |
| `moving_average` | batched window; at `K >= 4` the unrolled per-lane sum rebuilds into interleaved chains, adding {R}; below four terms every chained tree is the declared left fold, so the ordered form stays | {} or {R} by `K` | `fs_f32_solver` route audit (K=3 and K=8), `moving_average_f32.mlir` |
| `gain` | base graph | {} | `f32_gain_lms_aot`, three objects agree |
| `fir_filter`, `boundary = full` | mixed: guarded ordered edges, horizontal interior | {F} at each edge, {R} in the interior | `fp_fast_full_boundary_edge_aot`, executed edge skip |

The scalar `rms` spelling (`buffer` operand, bare `f32` result) lowers to
`dot(x, x)` followed by one division and one root, so it takes the `dot` row;
the FS route audit pins it separately (`rms scalar` cases).

What moves a route between rows:

- **Kernel stride.** `conv1d` bufferizes both modes to the same
  `ondsp.reduce_mac` over a kernel subview; correlation reads it forward at
  unit stride and is accepted, convolution's stride of −1 is refused. Same
  declaration, same operation, different mechanism.
- **Static length.** A reduction shorter than one vector block never enters the
  horizontal rewrite, so a short kernel takes the scalar route whatever its
  stride. The `conv1d` mode gate holds both modes at the same extents for
  exactly this reason.
- **Declared width.** `vector-bits=0` puts every exact route on the ordered
  scalar mechanism. A `fast` reduction keeps its multi-chain rebuild as scalar
  chains: chain count is ILP, not lane count, so the permission still buys the
  split dependence on a machine with no usable lanes.
- **Declared FMA capability.** `supports-vector-fma=true` moves every {R} row
  to {R, F}.

### The mixed route, and why the module union is not a record

A full-boundary `fir_filter` is the one route where a single declaration
reaches two mechanisms inside one function. Its output splits three ways: the
edge ranges, whose windows have out-of-range taps, keep the guarded ordered tap
loop and spend only F, while the interior emits `ondsp.reduce_mac` on a
unit-stride subview and reaches the horizontal rebuild, spending R.

The module-level summary reports `{F, R}` for this compilation. That is the
compilation's set, and it is claimed as nothing more: a reader wanting to know
what one route spent reads this ledger and the pass, not the attribute.

**Per-site attribution is deliberately not built.** A record keyed by site,
role, instance domain and branch condition was implemented and reverted. It was
correct and it answered the question honestly, but its cost was a plan threaded
through every floating-point lowering — four free-text strings per call site,
none of them checkable — so it taxed each future operation in proportion to the
capability work still ahead, to make an audit trail finer for a contract with
one exploiting transform. What the routes spend is stated here and gated by the
tests below; mechanical attribution waits for the same round as the rest of the
contract verification work, where a solver rather than hand-written bookkeeping
is the right instrument. Reintroducing it piecewise is not an improvement.

Two gates, because neither is sufficient.

`test/Permissions/fast_mixed_route_full_boundary.mlir` pins the generated
mechanism *cardinalities*: exactly one site reaches the rebuild, and under the
default term selection exactly two fused events exist and both are scalar.
Folding an edge into the vector route, recording the wrong permission at the
edge, and dropping the R record each redden the assertion that names that
property.

What counting cannot establish is which site owns which event. Deleting one
edge's fused event while adding one to the interior tail keeps the total at two,
so the gate bounds how many mechanisms were generated, and the per-site
selection record is what binds them to a route.

`test/Execution/fp_fast_full_boundary_edge_aot.mlir` executes the edge
semantics, because the structural pin cannot see them. This operation's
contract says an out-of-range tap performs no accumulator update rather than
contributing a zero term, and against a nonzero finite accumulator those agree
— every pre-existing full-boundary corpus is finite, so none of them could tell
the two apart even though their references model the skip correctly.

A materialized zero term is observable in exactly two ways, and there is no
third: a non-finite coefficient makes the product invalid, and a finite one
leaves only the sign of the zero. `0 * finite` cannot overflow, cannot produce a
non-zero subnormal, and cannot leave a rounding residue, while exception state
is unobservable under this contract and NaN payload is outside it. Both classes
are gated, one poison per trial so a pass attributes the skip to that value:

- **Non-finite.** An infinity, then a NaN, at a tap only the edge windows skip.
  Zero padding reaches `0 * poison` and returns NaN where the declared graph
  returns 1.5.
- **Signed zero.** `N = 1`, `K = 2`, one skipped tap after a valid term whose
  exact product underflows to `-0.0`. Skipping keeps `-0.0`; materializing the
  tap as `fma(+0.0, 1.0, -0.0)` returns `+0.0`.

The classes are independent, not belt-and-braces. Zero padding written as a
select plus one unconditional fused event leaves every operation count
unchanged, so the structural gate passes it and only the object rejects it; and
an implementation that guards non-finite coefficients while materializing finite
ones passes both non-finite trials, leaving the signed-zero trial as the only
one that rejects it.

## What the layers show

**Emitted is `{}` on every route.** Every permission a lowering spends is spent
by the compiler; none is handed to the backend.

**`fast` is genuinely unused on one route only.** `gain` is *semantically
inert*: its legal set is one graph, so there is nothing to spend on any
target. `moving_average` used to sit beside it as *operationally unused* —
its window sum is a designated reduction and **R** does apply, but no
transform rebuilt it. That distinction predicted its own end: the chained
window rebuild landed and the route now spends **R** at `K >= 4`, while
`gain` cannot change under any transform.

**Everything else consumes something.** The four routes that select a fused
chain spend **F**: choosing a fused event over a rounded product and an
addition is exactly what **F** authorizes, and the mutation that flips those
selections to separate multiply-add reddens their gates, which is what makes
the consumption observable rather than notional.

**Selection pins gate a policy, not a contract.** Where a route selects one
member, the object is pinned bitwise against that member. `fast` may legally
produce any member, so the purpose is that a transform which starts consuming
differently must redden the gate and re-justify itself.

**Mechanism evidence and reachability evidence are separate.**
`fp_fast_reduce_aot` shows that a `reduce_mac` becomes a term-conserving
horizontal schedule, under both term selections. It cannot show that a given
operation still forms the `reduce_mac` that reaches it, so
`test/Permissions/fast_route_reachability.mlir` pins that composition once per
adapter shape — whole contiguous memref, sliding unit-stride window, one
operand used twice, and the reversed subview whose refusal is the point.
Operations sharing a shape share the argument. The same file also pins the one
operation that presents no reduction shape at all: a matmul's column axis is
taken by the order-preserving batching, so nothing of it reaches the
horizontal route.

**What a compilation spent is recorded, not inferred.** A consumed permission
leaves no fast-math flag, so `consumeFastPermission` writes it down: an audit
attribute on the produced operation, unioned onto the module, and reported by
`ondrix-compile --emit=manifest`. The same source spends `{F}` with no declared
width, `{R}` at 256 bits, and `{R, F}` with a declared vector FMA — which is
the reachability layer above, made checkable per compilation.

## Format admission

Every executable floating-point path admits f32 and refuses anything else,
checked by `verifyExecutableFpFormat` at thirteen operation verifiers plus
`ondsp.reduce_mac`. `#ondsp.fp` itself stays format-parametric: widening is a
per-operation decision with its own evidence, not a property of the attribute.
The `.ox` frontend has no f64 spelling at all — its source types are Q15, Q31,
f32, and complex Q15 — so the refusal is reachable only from textual MLIR, and
that is where its negatives live.
