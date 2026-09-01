// RUN: ondrix-compile %S/Inputs/f32_cfft16.ox --emit=llvm --fft-loops | FileCheck %s --check-prefix=LOOPS
// RUN: ondrix-compile %S/Inputs/f32_cfft16.ox --emit=llvm | FileCheck %s --check-prefix=UNROLLED
// RUN: ondrix-compile %S/Inputs/f32_cfft16.ox --emit=manifest --fft-loops | FileCheck %s --check-prefix=MANIFEST
// RUN: ondrix-compile %S/Inputs/f32_cfft16.ox --emit=manifest | FileCheck %s --check-prefix=DEFAULT

// The code shape is an explicit schedule choice, not a target fact, so the
// source binding must reach it and the manifest must record it as one.

// Twenty products become four, because the stage loops read a table instead
// of carrying every twiddle as its own constant.
// LOOPS-COUNT-4: llvm.fmul
// LOOPS-NOT: llvm.fmul
// UNROLLED-COUNT-20: llvm.fmul
// UNROLLED-NOT: llvm.fmul

// The choice is recorded OUTSIDE declared_target_facts, because no target
// description determines it: a module holding both a size-8 and a size-64
// transform has no correct global value.
// MANIFEST: "declared_schedule_choices"
// MANIFEST-NEXT: "fft_lowering": "loops"
// MANIFEST: "declared_target_facts"
// MANIFEST-NOT: fft
// MANIFEST: convert-ondrix-to-ondsp{preserve-bufferizable-reductions=true fft-loops=true}
// DEFAULT: "fft_lowering": "unrolled"
// DEFAULT: convert-ondrix-to-ondsp{preserve-bufferizable-reductions=true fft-loops=false}
