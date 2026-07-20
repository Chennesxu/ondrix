# Implementation Status

Textual MLIR is currently the supported development and testing entry point.
Dialect contracts without a complete public consumer remain experimental and
may change while the numeric model is stabilized.

| Capability | Status |
| --- | --- |
| `ondrix`, `ondsp`, and `ortumcore` dialects | Implemented; contracts remain experimental |
| Typed conversion patterns and accumulator type conversion | Implemented |
| Generic scalar lowering | Signed-Q15 full product and signed-Q31 full/raw-high accumulator operations implemented; ordered rank-1 Q15/Q31 memref reductions implemented; rank-1 f32 reduction is partial |
| Generic Vector CPU lowering | Automatic unit-stride Q15/Q31 chunking, ordered saturating updates, and exact-modulo wrapping reduction implemented |
| Algorithm transforms | Opt-in specialization for static rank-1 FIR samples using immutable constant memref globals or proven complete unit-stride views: zero-tap elimination, exact-modulo symmetric full-product pairing, and symmetric saturating pairing when complete prefix-range analysis proves both schedules safe |
| Proof audit | Experimental optional JSON traces record constant saturating chunk-reassociation evidence; a read-only replay pass rederives immutable coefficient facts and the subject-bound prefix plan from the original IR, and rejects changes to proof-relevant semantics or trace data; traces are audit artifacts, not legality authority, an independent validator, or a stable certificate format |
| Full-output FIR | Experimental tensor-only destination-style `ondrix.fir_filter` for valid and zero-padded full boundaries; both expose output-axis tiling, with full tiles carrying an explicit global output origin; full lowering uses guarded scalar edges plus a Q15/Q31 Vectorized interior; immutable Q15/Q31 coefficients can activate prefix-proof-authorized horizontal Vector reduction after bufferization; ordered f32, dynamic shape guards, `K > N`, alias-safe tiled/untiled AOT execution, and direct bufferization are covered; same boundaries, public buffer semantics, fusion, and target-aware selection remain open |
| Streaming FIR | Experimental tensor-only `ondrix.fir_stream` with explicit chronological `K-1` history; ordered Q15/Q31/f32 generic scalar lowering, empty and short chunks, `K=1`, dynamic shape guards, multi-chunk state equivalence, and object+C execution are covered; an opt-in materialized concat decomposition reuses the existing Q15/Q31 Vector path; zero-copy buffer scheduling, public buffer semantics, reset/resampling policies, circular-buffer capabilities, and stable frontend binding remain open |
| Packed Q15 butterfly lowering | Experimental instruction selection only; no public emulator or ABI correctness claim |
| Object generation and C execution | Implemented for scalar and fixed-width Vector Q15/Q31 FIR-sample/full-output paths, ordered scalar Q15/Q31/f32 streaming chunks, and opt-in materialized Q15/Q31 streaming Vector reuse |
| Public OrtumCore emulation | Signed i40/frac30 saturating accumulator init and Q15 full-product MAC add/sub implemented through exact Ondsp expansion |
| Python-like `.ox` frontend | Planned |
| Stable public kernel ABI | Planned |

## Supported Q15 Slice

The executable path currently covers:

- signed Q15 values in signless `i16` storage;
- exact 16-by-16 full products;
- explicit signed i40 accumulators with per-update wrap or saturation;
- ordered rank-1 memref reductions;
- explicit export rounding and overflow;
- generic scalar and fixed-width Vector lowering;
- LLVM IR, PIC object generation, C linkage, and process execution.

The sample, full-output, and explicit-state streaming forms remain separate
algorithm contracts. Same-boundary FIR, streaming Vector scheduling, Q31 target
lowering, scalable Vector, a stable C ABI, and the `.ox` frontend remain outside
this supported Q15 slice.

## Supported Q31 Slice

The generic scalar and fixed-width Vector paths also cover:

- signed Q31 values in signless `i32` storage;
- exact full products accumulated in signed i64/frac62 state;
- signed raw-high products accumulated in signed i40/frac30 state;
- explicit Q31 export for full products and Q30 export for raw-high products;
- ordered saturating updates and exact-modulo wrapping reductions;
- LLVM IR, PIC object generation, C linkage, and process execution.

Q31 OrtumCore capability selection remains unsupported. Raw-high accumulation
does not implicitly rescale from Q30 to Q31.

## Public OrtumCore Emulator

The public emulator currently expands `ortumcore.acc_init`,
`ortumcore.mac_add`, and `ortumcore.mac_sub` to the exact signed-Q15 Ondsp
accumulator contract. The resulting Ondsp operations reuse the generic fixed
scalar finalizer. Other OrtumCore operations fail closed until they have a
typed equation and an executable public consumer.
