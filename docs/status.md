# Implementation Status

Textual MLIR remains the complete development and testing entry point. The
experimental `.ox` frontend currently covers Q15/Q31/f32 dot, FIR-sample,
dynamic/static valid FIR, and static full-boundary FIR over tensor values,
including immutable inline Q15/Q31 right operands for scalar reductions and
fixed full-output FIR, valid Q15/Q31/f32 convolution/correlation, static
four-/eight-point packed-Q15 CFFT, Q15 streaming FIR, and one static-section
Q15 fixed DF-II SOS profile.
Dialect and source contracts may change while the numeric model is stabilized.

| Capability | Status |
| --- | --- |
| `ondrix`, `ondsp`, and `ortumcore` dialects | Implemented; contracts remain experimental |
| Typed conversion patterns and accumulator type conversion | Implemented |
| Generic scalar lowering | Signed-Q15 full product with signed frac30 accumulator widths of at least 32 bits and signed-Q31 full/raw-high accumulator operations implemented; ordered rank-1 Q15/Q31 memref reductions implemented; rank-1 f32 reduction is partial |
| Generic Vector CPU lowering | Automatic unit-stride Q15/Q31 chunking, ordered saturating updates, and exact-modulo wrapping reduction implemented |
| Algorithm transforms | Opt-in specialization for static rank-1 FIR samples using immutable constant memref globals or proven complete unit-stride views: zero-tap elimination, exact-modulo symmetric full-product pairing, and symmetric saturating pairing when complete prefix-range analysis proves both schedules safe |
| Proof audit | Experimental optional JSON traces record constant saturating chunk-reassociation evidence; a read-only replay pass rederives immutable coefficient facts and the subject-bound prefix plan from the original IR, and rejects changes to proof-relevant semantics or trace data; traces are audit artifacts, not legality authority, an independent validator, or a stable certificate format |
| Full-output FIR | Experimental tensor-only destination-style `ondrix.fir_filter` for valid and zero-padded full boundaries; both expose output-axis tiling, with full tiles carrying an explicit global output origin; full lowering uses guarded scalar edges plus a Q15/Q31 Vectorized interior; immutable Q15/Q31 coefficients can activate prefix-proof-authorized horizontal Vector reduction after bufferization; ordered f32, dynamic shape guards, `K > N`, alias-safe tiled/untiled AOT execution, and direct bufferization are covered; same boundaries, public buffer semantics, fusion, and target-aware selection remain open |
| FIR decimation | Experimental tensor-only `ondrix.fir_decimate` defines valid phase-zero decimation with static factor, increasing tap order, and result length `floor((N-K)/D)+1`; one factor-two signed-Q15/full+i40 profile has generic Ondsp lowering and object+C differential execution; interpolation, nonzero phase, polyphase transforms, f32/Q31, Vector, frontend, and target consumers remain unsupported |
| Convolution/correlation | Experimental tensor-only `ondrix.conv1d` preserves distinct valid one-dimensional convolution and correlation identities with explicit kernel orientation and increasing-window update order; Q15 and f32 generic tensor lowering, direct bufferization, dynamic shape guards, LLVM object+C differential execution, unit-stride correlation Vector reuse, and reverse-stride convolution scalar fallback are covered; the `.ox` frontend exposes matching valid tensor forms; full/same padding, stride, dilation, multidimensional/grouped forms, complex conjugation, and target selection remain unsupported |
| Streaming FIR | Experimental tensor-only `ondrix.fir_stream` with explicit chronological `K-1` history; ordered Q15/Q31/f32 generic scalar lowering, empty and short chunks, `K=1`, dynamic shape guards, multi-chunk state equivalence, and object+C execution are covered; an opt-in materialized concat decomposition reuses the existing Q15/Q31 Vector path; `.ox` exposes one dynamic-chunk/static-tap Q15 profile with inferred exact accumulation and two tensor results; zero-copy buffer scheduling, broader source profiles, reset/resampling policies, circular-buffer capabilities, and stable ABI remain open |
| Recursive filters | Experimental tensor-only `ondrix.sos_filter_tdf2` defines an explicit f32 transposed-direct-form-II cascade; experimental `ondrix.sos_filter_df2_fixed` separately defines a signed uniform-Q DF-II cascade for Q15/full+i40/frac30 and Q31/full+i64/frac62 with independent state/output export policies; both have dynamic section guards, split/empty-chunk state equivalence, generic lowering, and object+C differential execution; `.ox` exposes one static-section Q15 DF-II profile with explicit accumulator and state/output policies plus whole/split execution; nonuniform fixed scaling, raw-high products, Vector lowering, public buffer semantics, target selection, broader source profiles, and stable state ABI remain unsupported |
| Packed Q15 butterfly | One closed scalar/fixed-Vector `i32` profile is executable: signed-Q15 complex values use imag-high/real-low packing, exact i33/Q30 cross-products, explicit nearest-even saturating product and output requantization, generic elementwise lowering, and object+C differential execution; broader complex profiles, frontend butterfly syntax, and OrtumCore selection remain unsupported |
| Packed Q15 CFFT | Experimental static four- and eight-point forward and inverse radix-2 DIT transforms implemented over matching `tensor<Nxi32>` values with natural input/output order, recursive even/odd stage decomposition, frozen Q15 twiddle tables, explicit per-stage scaling, generic scalar lowering, and object+C differential execution; an opt-in conversion mode batches independent size-4/8 combine-stage butterflies into fixed-length Vector arithmetic without crossing stage or requantization boundaries, and agrees bit-for-bit with scalar/reference execution; inverse transforms use conjugated twiddles and the per-stage shifts provide `1/N` normalization; an independent opt-in scalar-finalizer mode derives exact finite-precision `+1` and `-j` identities from forward constant twiddles while other constants and runtime twiddles retain generic i33 products; the `.ox` frontend exposes the bounded profiles as `cfft` and `icfft` over `tensor[complex_q15,4|8]`; other sizes and target execution remain unsupported |
| Object generation and C execution | Implemented for scalar and fixed-width Vector Q15/Q31 FIR-sample/full-output paths, factor-two Q15 FIR decimation, valid Q15/f32 convolution/correlation, ordered scalar Q15/Q31/f32 streaming chunks, opt-in materialized Q15/Q31 streaming Vector reuse, the experimental f32 TDF-II plus uniform-Q fixed DF-II SOS cascades, the closed packed-Q15 scalar/Vector butterfly and static four-/eight-point scalar/Vector forward/inverse CFFT profiles, and the `.ox` Q15/Q31/f32 dot/FIR, Q15 streaming FIR/fixed SOS, and packed-Q15 CFFT/ICFFT source slices, including constexpr fixed FIR specialization |
| Public OrtumCore emulation | Signed i40/frac30 saturating accumulator init and Q15 full-product MAC add/sub implemented through exact Ondsp expansion |
| Python-like `.ox` frontend | Experimental standalone Lexer/Parser/AST/Sema/MLIRGen pipeline for rank-1 Q15/Q31/f32 dot, FIR-sample, tensor-value valid/full FIR, valid convolution/correlation, static four-/eight-point packed-Q15 forward/inverse CFFT, one packed-Q15 butterfly, dynamic-chunk/static-tap Q15 streaming FIR, and one rank-2 static-section Q15 fixed DF-II SOS profile; statically bounded Q15 feed-forward reductions may infer a sufficient exact accumulator width, while recursive SOS retains explicit i40 update and separate state/output export policies; fixed accumulator/export policies, ordered f32 contraction, convolution kernel orientation, chronological streaming state, and the closed `complex_q15` stage-scaling profiles are explicit, source locations reach MLIR, diagnostics are source-aware, and object+C execution is covered; dynamic full boundary/FFT planning, convolution padding/stride/dilation, indexing, loops, multiple kernels, mutable output buffers, broader stateful profiles, and stable source/ABI contracts remain planned |
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
lowering, scalable Vector, and a stable C ABI remain outside this supported Q15
slice. The experimental `.ox` Q15 dot and FIR-sample forms are narrow source
entries into the same scalar contract, not a stable language surface.

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
