# ondrix

ondrix is an MLIR-based AOT compiler framework for portable embedded DSP
kernels.

The project explores a layered IR stack for embedded DSP compilation. Kernels
are intended to be written in an Ondrix kernel language and lowered by a
frontend into the `ondrix` MLIR dialect. The project borrows the idea of
separating high-level kernel intent from progressively lower IR, but it is not a
GPU execution framework and does not use JIT compilation. The target flow is:

```text
Ondrix kernel language
  -> frontend
  -> ondrix
  -> ondsp
  -> generic LLVM path or ortumcore
  -> LLVM dialect / LLVM IR
  -> object file, static library, or firmware integration
```

Textual MLIR remains the primary test and debugging format for the first-stage
compiler implementation.

## Architecture

`ondrix` is the high-level DSP algorithm intent dialect. It represents kernels
such as FIR, dot product, correlation, quantization, and complex butterfly
patterns without exposing target instruction names.

`ondsp` is the target-independent DSP numeric semantic dialect. It models
fixed-point policy, floating-point policy, accumulator state, complex packing,
rounding, saturation, shifting, MAC operations, and block-friendly DSP structure.

`ortumcore` is the first target-specific DSP semantic dialect. It models
hardware-aware DSP operations at the IR level and lowers to LLVM dialect call
stubs and AOT support stubs in this public repository.

## First-Stage Scope

The first-stage implementation focuses on a small, testable AOT baseline:

- q15 FIR / dot path:
  `ondrix.fir` -> `ondsp.reduce_mac` / `ondsp.mac` ->
  `ortumcore` MAC semantic ops -> LLVM dialect call stubs.
- f32 FIR / dot path:
  `ondrix.fir` -> `ondsp.reduce_mac` -> generic arith/scf/LLVM-friendly IR ->
  LLVM dialect / LLVM IR.
- q15 packed-complex butterfly path:
  `ondrix.butterfly` or `ondrix.fft_stage` -> `ondsp.cx_butterfly` /
  `ondsp.fft_stage` -> `ortumcore` complex/FFT semantic ops -> LLVM dialect call
  stubs.

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

Local development in this workspace uses:

```sh
cmake -G Ninja \
  -S /home/xuch/research/Ondrix/ondrix \
  -B /home/xuch/research/Ondrix/ondrix/build \
  -DMLIR_DIR=/home/xuch/research/Ondrix/llvm-project/build/lib/cmake/mlir \
  -DLLVM_DIR=/home/xuch/research/Ondrix/llvm-project/build/lib/cmake/llvm \
  -DLLVM_EXTERNAL_LIT=/home/xuch/research/Ondrix/llvm-project/build/bin/llvm-lit
```

Build and test:

```sh
ninja -C /path/to/ondrix/build ondrix-opt
ninja -C /path/to/ondrix/build check-ondrix
```

## Reference Material Policy

Some design references may exist outside this repository in a local, read-only
reference directory. They are not part of the open-source project and must not
be copied into this repository. Public documentation and source comments should
describe only original, abstract design decisions and short semantic summaries.
