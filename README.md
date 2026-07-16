# ondrix

ondrix is a language and compiler framework for writing portable embedded DSP
kernels. The aim of the project is to let developers describe common DSP kernels
in a simple Python-like language instead of hand-writing and maintaining large
assembly kernel libraries, while still producing AOT code suitable for embedded
firmware or static-library integration.

The planned source language is Python-like and lowers into MLIR. The compiler
uses a layered IR stack to separate high-level kernel intent,
target-independent DSP numeric semantics, and target-specific DSP operations.

The target flow is:

```text
ondrix language
  -> frontend
  -> ondrix dialect
  -> ondsp dialect
  -> generic scalar, generic vector, or ortumcore AOT-stub path
  -> LLVM dialect / LLVM IR
  -> object file, static library, or firmware integration
```

## Architecture

`ondrix` is the high-level DSP algorithm intent dialect. It represents kernels
such as FIR, dot product, correlation, quantization, and complex butterfly
patterns without exposing target instruction names.

`ondsp` is the target-independent DSP numeric semantic dialect. It models
fixed-point policy, floating-point policy, accumulator state, complex packing,
rounding, saturation, shifting, MAC operations, and block-friendly DSP structure.

`ortumcore` is the first target-specific DSP semantic dialect. It models
hardware-aware DSP operations at the IR level. Public emulation and LLVM call
stub lowering are planned consumers of this dialect.

The generic scalar path is intended for correctness testing and fallback
compilation. The generic vector path is intended for portable CPU code
generation through LLVM. The ortumcore path provides an abstract target semantic
layer without exposing private backend details.

## Current Status

| Capability | Status |
| --- | --- |
| `ondrix`, `ondsp`, and `ortumcore` dialects | Implemented, contracts still experimental |
| Typed conversion patterns and accumulator type conversion | Implemented |
| Generic scalar lowering | Implemented for explicit signed Q15 accumulator operations and ordered rank-1 Q15 memref reductions; partial for rank-1 f32 reductions |
| Packed Q15 butterfly lowering | Experimental instruction selection only; no public emulator or ABI correctness claim |
| Generic Vector CPU lowering | Automatic unit-stride Q15 memref chunking, ordered saturating updates, and exact-modulo wrap reduction implemented |
| Public OrtumCore emulation / LLVM stubs | Planned |
| Python-like `.ox` frontend | Planned |
| Object generation and C ABI execution tests | Implemented for scalar and automatic Vector Q15 FIR-sample/dot paths, including dynamic lengths, tails, overflow, and scalar fallback; a stable public kernel ABI is planned |

Textual MLIR is currently the supported development and testing entry point.
Operations without a complete public consumer remain experimental and may
change while the numeric contracts are being stabilized.

## Executable Q15 Contract

The first executable fixed-point slice uses signless `i16` storage interpreted
as signed Q15 (`frac = 15`) and an explicit accumulator such as:

```mlir
!ondsp.acc<storage = i40, frac = 30, signed,
             update_overflow = saturate>
```

`ondsp.mac` computes an exact signed 16-by-16 full product, adds it to the
mathematical accumulator value, and then applies the accumulator's per-update
`wrap` or `saturate` policy at the declared storage width. `ondsp.reduce_mac`
is the increasing-index left fold of the same update over two equal-length
rank-1 operands and an explicit initial accumulator. Reassociation is not part
of this contract; every Vector or target lowering must prove equivalence before
changing the update order.

Leaving the accumulator domain requires `ondsp.acc_export`, which explicitly
states destination numeric format, rounding, and destination overflow. The
generic scalar implementation is checked against an independent integer
reference through real LLVM IR, object generation, C linkage, and process
execution, including 40-bit overflow boundaries and dynamic-memref FIR/dot
cases.

The Vector path automatically splits unit-stride rank-1 memref reductions into
fixed-width chunks and a scalar tail. Q15 products use Vector dialect
operations. Saturating accumulators apply those products in increasing lane
order. Wrapping accumulators may use a horizontal vector reduction because the
Ondsp legality query classifies fixed-width modular addition as exactly
reassociable. Memrefs without a statically known unit minor stride retain the
generic scalar fallback.

## LLVM/MLIR Baseline

ondrix is developed against external LLVM/MLIR 17.0.6:

- LLVM tag: `llvmorg-17.0.6`
- LLVM hash: `6009708b4367171ccdbf4b5905cb6a803753fe18`

MLIR does not provide a stable C++ API across major versions, so users should
build against the same LLVM/MLIR baseline unless they are intentionally doing a
version migration.

## Build

Configure with an external LLVM/MLIR build:

```sh
cmake -G Ninja \
  -S /path/to/ondrix \
  -B /path/to/ondrix/build \
  -DMLIR_DIR=/path/to/llvm-project/build/lib/cmake/mlir \
  -DLLVM_DIR=/path/to/llvm-project/build/lib/cmake/llvm \
  -DLLVM_EXTERNAL_LIT=/path/to/llvm-project/build/bin/llvm-lit
```

Build and test:

```sh
ninja -C /path/to/ondrix/build ondrix-opt
ninja -C /path/to/ondrix/build check-ondrix
```
