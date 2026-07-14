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
| Generic scalar lowering | Partial: rank-1 f32 reduction only |
| Generic Vector CPU lowering | Planned |
| Public OrtumCore emulation / LLVM stubs | Planned |
| Python-like `.ox` frontend | Planned |
| Object generation and C ABI execution tests | Planned |

Textual MLIR is currently the supported development and testing entry point.
Operations without a complete public consumer remain experimental and may
change while the numeric contracts are being stabilized.

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
