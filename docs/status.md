# Implementation Status

Textual MLIR is currently the supported development and testing entry point.
Dialect contracts without a complete public consumer remain experimental and
may change while the numeric model is stabilized.

| Capability | Status |
| --- | --- |
| `ondrix`, `ondsp`, and `ortumcore` dialects | Implemented; contracts remain experimental |
| Typed conversion patterns and accumulator type conversion | Implemented |
| Generic scalar lowering | Signed-Q15 accumulator operations and ordered rank-1 Q15 memref reductions implemented; rank-1 f32 reduction is partial |
| Generic Vector CPU lowering | Automatic unit-stride Q15 chunking, ordered saturating updates, and exact-modulo wrapping reduction implemented |
| Packed Q15 butterfly lowering | Experimental instruction selection only; no public emulator or ABI correctness claim |
| Object generation and C execution | Implemented for scalar and fixed-width Vector Q15 FIR-sample and dot paths |
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
input window. Full-output and streaming FIR, executable Q31 arithmetic,
scalable Vector, a stable C ABI, and the `.ox` frontend are outside this
supported slice.
