# f32 contract evidence ledger

One row per operation and declared contract across the thirteen real,
non-transform operations of the f32 profile. The point of the table is not to
show that everything is covered — it is to make visible which permissions are
actually spent, and where a declaration is admitted with no executed evidence
behind it.

## Facets

A declaration is not a schedule. Six things are distinct and the table keeps
them apart:

- **Base graph** — the event graph the operation's description states
  literally, in declared order.
- **Legal set** — the graphs derivable from the base graph using only the
  rewrites the declaration permits. `off` and `fma` each permit none, so their
  legal set is a single graph and they are exact contracts: bitwise
  reproducible for non-NaN outputs, NaN class preserved, signed zeros and
  infinities bitwise.
- **Selected** — the one member the lowering builds. This is a compiler
  policy, not a language rule; nothing forbids a different member.
- **Used** — the permissions the compiler spent to reach the selected graph.
- **Emitted** — the permissions left on the operations that reach the audit
  point. Spending a permission and emitting it are opposites: an emitted flag
  hands the choice to LLVM instead of making it. Emitted is `{}` everywhere in
  the table, and `test/Permissions/fp_contract_permission_audit.mlir` gates
  that on the translated LLVM IR.
- **Executed gate** — the object-level evidence. An exact contract is pinned
  bitwise against an independent reference; a relaxed one cannot be, so its
  evidence is term conservation, class preservation, and divergence from the
  ordered chain.

The audit point is the translated `.ll`: this flow runs no LLVM middle end, so
permissions are final there. It is *not* the final point for the realized
event graph, because `llc` still consumes fast-math flags — on the pinned
toolchain a `reassoc` on `llvm.fma` is enough for the backend to de-fuse it
(`test/Target/fp_permission_fmf.ll`). That is why emitted is empty rather than
merely bounded.

`fast` permits two rewrites: `ReassociateReductionTerms` (regroup a
reduction's additive tree) and `FuseMultiplyAdd` (select a fused event for a
term). Below, **R** and **F**.

## Ledger

| Operation | Contract | Legal set | Selected | Used | Executed gate |
| --- | --- | --- | --- | --- | --- |
| `dot` | off | base | — | {} | `fp_scalar_dot_aot`, `ox_f32_dot_aot` |
| `dot` | fma | fused chain | — | {} | `fp_scalar_dot_aot` |
| `dot` | fast | regroupings × per-term fusion | lane partition, separate products | {R} | `fp_scalar_dot_aot`, `fp_fast_reduce_aot` |
| `fir` | off | base | — | {} | `ox_f32_fir_aot` |
| `fir` | fma | fused chain | — | {} | `ox_f32_fir_aot` |
| `fir` | fast | regroupings × per-term fusion | lane partition, separate products | {R} | mechanism only (`fp_fast_reduce_aot`) |
| `fir_filter` | off | base | — | {} | `fp_filter_output_batch_aot` |
| `fir_filter` | fma | fused chain | — | {} | `fp_filter_output_batch_aot` |
| `fir_filter` | fast | regroupings × per-term fusion | lane partition, separate products | {R} | mechanism only |
| `conv1d` | off | base | — | {} | none |
| `conv1d` | fma | fused chain | — | {} | none |
| `conv1d` | fast | regroupings × per-term fusion | ordered scalar | {} | none |
| `matmul` | off | base | — | {} | `f32_matmul_rms_aot` |
| `matmul` | fma | fused chain | — | {} | `f32_matmul_rms_aot` |
| `matmul` | fast | per-element regroupings × fusion | lane partition, separate products | {R} | mechanism only |
| `rms` | off | base | — | {} | `f32_matmul_rms_aot` |
| `rms` | fma | fused chain | — | {} | `f32_matmul_rms_aot` |
| `rms` | fast | regroupings of the square sum × fusion | lane partition, separate products | {R} | mechanism only |
| `moving_average` | off | base | — | {} | `f32_unary_aot` |
| `moving_average` | fma | fused chain | — | {} | none |
| `moving_average` | fast | window regroupings × fusion | ordered scalar | {} | none |
| `dct` | off | base | — | {} | `f32_unary_aot` |
| `dct` | fma | fused chain | — | {} | `f32_unary_aot` |
| `dct` | fast | per-row regroupings × fusion | ordered scalar | {} | none |
| `fir_decimate` | off | base | — | {} | `f32_resampling_aot` |
| `fir_decimate` | fma | fused chain | — | {} | `f32_resampling_aot` |
| `fir_decimate` | fast | regroupings × per-term fusion | lane partition, separate products | {R} | mechanism only |
| `fir_interpolate` | off | base | — | {} | `f32_resampling_aot`, skip witnesses |
| `fir_interpolate` | fma | fused chain | — | {} | `f32_resampling_aot`, skip witnesses |
| `fir_interpolate` | fast | regroupings × per-term fusion | ordered scalar | {} | none |
| `gain` | off | base | — | {} | `f32_gain_lms_aot` |
| `gain` | fma | base (nothing to fuse) | — | {} | `f32_gain_lms_aot` |
| `gain` | fast | base (nothing to regroup or fuse) | base | {} | `f32_gain_lms_aot`, three objects agree bitwise |
| `lms` | off | base | — | {} | `f32_gain_lms_aot` |
| `lms` | fma | fused chain | — | {} | `f32_gain_lms_aot` |
| `lms` | fast | tap regroupings × per-term fusion | ordered scalar | {} | none |
| `goertzel` | off | base | — | {} | `f32_goertzel_aot` |
| `goertzel` | fma | fused recursion | — | {} | `f32_goertzel_aot` |
| `goertzel` | fast | exactly two: the off and fma graphs | the fma graph | {F} | `f32_goertzel_aot`, bitwise against fma plus a `.ll` permission pin |

## What the table shows

**Emitted is `{}` in all thirty-nine rows.** Every permission a lowering
spends is spent by the compiler; none is handed to the backend.

**`fast` currently buys one thing.** Six operations reach the term-conserving
horizontal reduction and spend `ReassociateReductionTerms`; `goertzel` spends
`FuseMultiplyAdd` at its single multiply-add site; `gain` has nothing to
spend. The remaining five declare `fast` and get the ordered schedule. That is
sound — a permission may go unused — but it means the declaration is currently
inert there, and a reader should not infer a faster object from it.

**Executed `fast` coverage is thinner than admitted `fast` surface.** Four
operations gate `fast` at the object level. Six more rely on
`fp_fast_reduce_aot`, which gates the shared reduction mechanism rather than
the operation that reaches it. Five have no `fast` object evidence at all,
which is acceptable only because their selected graph is the ordered one they
would have produced under `off`.

**`conv1d` has no f32 object gate on any contract.** It verifies, lowers, and
is covered by conversion checks only.

## Format admission

Every executable floating-point path in the catalog admits f32 and refuses
anything else, checked by `verifyExecutableFpFormat` at thirteen operation
verifiers plus `ondsp.reduce_mac`. `#ondsp.fp` itself stays format-parametric:
widening is a per-operation decision with its own evidence, not a property of
the attribute. The `.ox` frontend has no f64 spelling at all — its source
types are Q15, Q31, f32, and complex Q15 — so the refusal is reachable only
from textual MLIR, and that is where its negatives live.
