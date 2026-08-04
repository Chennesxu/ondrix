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
  operation carries no flag; regrouping across lanes spends **R**.
- **Emitted** — permissions left on the operations that reach the audit point.
  Spending and emitting are opposites: an emitted flag hands the choice to
  LLVM instead of making it. Emitted is `{}` everywhere, gated by
  `test/Permissions/fp_contract_permission_audit.mlir` on the translated IR.
- **Executed gate** — object-level evidence. An exact contract is pinned
  bitwise against an independent reference. A relaxed result is not pinned; its
  evidence is term conservation on a sub-domain where every derivable
  regrouping agrees.

The audit point is the translated `.ll`: this flow runs no LLVM middle end, so
permissions are final there. It is *not* the final point for the realized event
graph, because `llc` still consumes fast-math flags — a `reassoc` on `llvm.fma`
is enough for the X86 backend without +fma to de-fuse it, while AArch64 and the
32-bit ARM DSP targets keep it fused even with no fused instruction to keep
(`test/Target/fp_permission_fmf_*.ll`). Which graph runs is a per-backend
expansion policy that a delegated permission cannot bound, which is why emitted
is empty rather than merely bounded.

`fast` permits **R** (`RebuildReductionTree`) and **F** (`FuseMultiplyAdd`).

**R** is a tree rebuild, not reparenthesization: the target may be any binary
addition tree whose leaves are a bijection onto the source reduction's indexed
terms — each exactly once, operands unchanged, no identity introduced. That
covers permuting the terms as well as regrouping them, and the wider form is
what a lane partition needs. Lane `i` pairs term `i` with term `i + W`, and
reparenthesization alone preserves leaf order, so associativity by itself would
not authorize the schedule this compiler builds. The difference is observable:
over `[1e8, 1.0, 0, …, -1e8, …]` at N=16, W=8 the source leaf order gives 0 and
the lane partition gives 1.

**F** selects a fused multiply-add event for a term in place of a rounded
product followed by an addition.

## Layer 1 — language: what each operation's `fast` admits

| Operation | Legal set under `fast` |
| --- | --- |
| `gain` | one graph. A lone product has no addend to fuse and no tree to regroup — the only operation whose three declarations denote the same events. |
| `moving_average` | the window sum is a reduction tree, so **R** applies for `K >= 2`; no product, so **F** never does. The rebuilt trees first differ in value at `K = 3`, since a two-term sum is commutative up to NaN payload. |
| `dot`, `fir`, `fir_filter`, `conv1d`, `matmul`, `rms`, `dct`, `fir_decimate`, `fir_interpolate`, `lms` | products plus an additive tree: both **R** and **F** apply. |
| `goertzel` | exactly two graphs. A recursion has no reduction to regroup, so only **F** applies, at one multiply-add site. |
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
| `dot`, `fir`, `fir_filter`, `matmul`, `rms`, `fir_decimate` | horizontal, separate | {R} | mechanism only (`fp_fast_reduce_aot`) |
| `conv1d` correlation, K ≥ W | horizontal, separate | {R} | `f32_conv1d_aot`, term conservation |
| `conv1d` convolution | scalar fused | {F} | `f32_conv1d_aot`, selection pinned |
| `dct`, `lms`, `fir_interpolate` | scalar fused | {F} | selection pinned |
| `goertzel` | scalar fused | {F} | `f32_goertzel_aot`, bitwise against `fma` plus a `.ll` permission pin |
| `moving_average` | ordered scalar | {} | `f32_unary_aot`, association witness |
| `gain` | base graph | {} | `f32_gain_lms_aot`, three objects agree |

What moves a route between rows:

- **Kernel stride.** `conv1d` bufferizes both modes to the same
  `ondsp.reduce_mac` over a kernel subview; correlation reads it forward at
  unit stride and is accepted, convolution's stride of −1 is refused. Same
  declaration, same operation, different mechanism.
- **Static length.** A reduction shorter than one vector block never enters the
  horizontal rewrite, so a short kernel takes the scalar route whatever its
  stride. The `conv1d` mode gate holds both modes at the same extents for
  exactly this reason.
- **Declared width.** `vector-bits=0` puts every route on the ordered scalar
  mechanism.
- **Declared FMA capability.** `supports-vector-fma=true` moves every {R} row
  to {R, F}.

## What the layers show

**Emitted is `{}` on every route.** Every permission a lowering spends is spent
by the compiler; none is handed to the backend.

**`fast` is genuinely unused on two routes only**, and for different reasons.
`gain` is *semantically inert*: its legal set is one graph, so there is nothing
to spend on any target. `moving_average` is *operationally unused*: its window
sum is a reduction tree and **R** does apply, but the lowering builds the
declared association and no transform regroups it. The distinction matters —
the second can change when a transform lands, the first cannot.

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
adapter shape — whole contiguous memref, sliding unit-stride window, matrix row
against a transposed pack, one operand used twice, and the reversed subview
whose refusal is the point. Operations sharing a shape share the argument.

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
