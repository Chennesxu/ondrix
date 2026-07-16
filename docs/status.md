# Implementation Status

Textual MLIR is currently the supported development and testing entry point.
Dialect contracts without a complete public consumer remain experimental and
may change while the numeric model is stabilized.

| Capability | Status |
| --- | --- |
| `ondrix`, `ondsp`, and `ortumcore` dialects | Implemented; contracts remain experimental |
| Typed conversion patterns and accumulator type conversion | Implemented |
| Generic scalar lowering | Signed-Q15 full product and signed-Q31 full/raw-high accumulator operations implemented; ordered rank-1 Q15/Q31 memref reductions implemented; rank-1 f32 reduction is partial |
| Generic Vector CPU lowering | Automatic unit-stride Q15 chunking, ordered saturating updates, and exact-modulo wrapping reduction implemented |
| Packed Q15 butterfly lowering | Experimental instruction selection only; no public emulator or ABI correctness claim |
| Object generation and C execution | Implemented for scalar and fixed-width Vector Q15 FIR-sample/dot paths and scalar Q31 numeric operations |
| Public OrtumCore emulation or LLVM stubs | Planned |
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

The current FIR operation represents one output sample over a preconstructed
input window. Full-output and streaming FIR, Q31 Vector and target lowering,
scalable Vector, a stable C ABI, and the `.ox` frontend are outside this
supported Q15 slice.

## Supported Q31 Scalar Slice

The generic scalar path also covers:

- signed Q31 values in signless `i32` storage;
- exact full products accumulated in signed i64/frac62 state;
- signed raw-high products accumulated in signed i40/frac30 state;
- explicit Q31 export for full products and Q30 export for raw-high products;
- LLVM IR, PIC object generation, C linkage, and process execution.

Q31 Vector lowering and Q31 OrtumCore capability selection remain unsupported.
Raw-high accumulation does not implicitly rescale from Q30 to Q31.
