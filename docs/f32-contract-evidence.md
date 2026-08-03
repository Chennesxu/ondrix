# f32 contract evidence ledger

One row per operation and declared contract across the thirteen real,
non-transform operations of the f32 profile — forty rather than thirty-nine,
because `conv1d` spends a permission under `fast` in one mode and nothing in
the other. The point of the table is not to
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
| `conv1d` | off | base | — | {} | `f32_conv1d_aot`, both modes |
| `conv1d` | fma | fused chain | — | {} | `f32_conv1d_aot`, both modes |
| `conv1d` correlation | fast | regroupings × per-term fusion | lane partition, separate products | {R} | `f32_conv1d_aot`, term conservation |
| `conv1d` convolution | fast | regroupings × per-term fusion | fused chain | {} | `f32_conv1d_aot`, selection pinned |
| `matmul` | off | base | — | {} | `f32_matmul_rms_aot` |
| `matmul` | fma | fused chain | — | {} | `f32_matmul_rms_aot` |
| `matmul` | fast | per-element regroupings × fusion | lane partition, separate products | {R} | mechanism only |
| `rms` | off | base | — | {} | `f32_matmul_rms_aot` |
| `rms` | fma | fused chain | — | {} | `f32_matmul_rms_aot` |
| `rms` | fast | regroupings of the square sum × fusion | lane partition, separate products | {R} | mechanism only |
| `moving_average` | off | base | — | {} | `f32_unary_aot` |
| `moving_average` | fma | base (no product to fuse) | — | {} | `f32_unary_aot` |
| `moving_average` | fast | base (nothing to regroup or fuse) | base | {} | `f32_unary_aot`, three objects agree |
| `dct` | off | base | — | {} | `f32_unary_aot` |
| `dct` | fma | fused chain | — | {} | `f32_unary_aot` |
| `dct` | fast | per-row regroupings × fusion | fused chain | {} | `f32_unary_aot`, selection pinned |
| `fir_decimate` | off | base | — | {} | `f32_resampling_aot` |
| `fir_decimate` | fma | fused chain | — | {} | `f32_resampling_aot` |
| `fir_decimate` | fast | regroupings × per-term fusion | lane partition, separate products | {R} | mechanism only |
| `fir_interpolate` | off | base | — | {} | `f32_resampling_aot`, skip witnesses |
| `fir_interpolate` | fma | fused chain | — | {} | `f32_resampling_aot`, skip witnesses |
| `fir_interpolate` | fast | regroupings × per-term fusion | fused chain | {} | `f32_resampling_aot`, selection pinned |
| `gain` | off | base | — | {} | `f32_gain_lms_aot` |
| `gain` | fma | base (nothing to fuse) | — | {} | `f32_gain_lms_aot` |
| `gain` | fast | base (nothing to regroup or fuse) | base | {} | `f32_gain_lms_aot`, three objects agree bitwise |
| `lms` | off | base | — | {} | `f32_gain_lms_aot` |
| `lms` | fma | fused chain | — | {} | `f32_gain_lms_aot` |
| `lms` | fast | tap regroupings × per-term fusion | fused chain | {} | `f32_gain_lms_aot`, selection pinned |
| `goertzel` | off | base | — | {} | `f32_goertzel_aot` |
| `goertzel` | fma | fused recursion | — | {} | `f32_goertzel_aot` |
| `goertzel` | fast | exactly two: the off and fma graphs | the fma graph | {F} | `f32_goertzel_aot`, bitwise against fma plus a `.ll` permission pin |

## What the table shows

**Emitted is `{}` in all forty rows.** Every permission a lowering
spends is spent by the compiler; none is handed to the backend.

**`fast` buys one thing, and only where the schedule can reach it.** Seven
operations spend `ReassociateReductionTerms` in the term-conserving horizontal
reduction; `goertzel` spends `FuseMultiplyAdd` at its single multiply-add site;
`gain` and `moving_average` have nothing to spend, their legal set being a
single graph. The remaining four declare `fast`, could in principle use it, and
get the fused ordered chain anyway. That is sound — a permission may go unused
— but the declaration is inert there and no faster object should be inferred
from it.

**Reachability, not the operation, decides.** `conv1d` is the clearest case:
both modes bufferize to the same `ondsp.reduce_mac` over a kernel subview, but
correlation reads it forward at unit stride and is accepted, while
convolution's stride of −1 is refused. The same declaration on the same
operation therefore spends a permission in one mode and nothing in the other.
Anything that would make a reversed subview vectorizable changes that row.

**Every inert `fast` is now pinned to the member the lowering selects.** Those
gates assert a compiler policy, not a contract: `fast` may legally produce any
member of its set, so a transform that starts consuming at one of those sites
must redden its gate and re-justify itself rather than change the object
silently. Relaxed results stay unpinned — the consuming legs are checked for
term conservation on an integer sub-domain where every derivable regrouping is
exact.

**Admission follows the legal set.** A legal set of one graph is provably inert
and admitted; a larger set is admitted when a derivability gate exists for what
is selected. `sos_filter_tdf2` fails both tests — fusable multiply-adds in a
recurrence with no recurrence gate — and refuses `fast` at the verifier.

## Format admission

Every executable floating-point path in the catalog admits f32 and refuses
anything else, checked by `verifyExecutableFpFormat` at thirteen operation
verifiers plus `ondsp.reduce_mac`. `#ondsp.fp` itself stays format-parametric:
widening is a per-operation decision with its own evidence, not a property of
the attribute. The `.ox` frontend has no f64 spelling at all — its source
types are Q15, Q31, f32, and complex Q15 — so the refusal is reachable only
from textual MLIR, and that is where its negatives live.
