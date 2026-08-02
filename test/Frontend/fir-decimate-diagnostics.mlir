// RUN: not ondrix-compile %S/Inputs/invalid_fir_decimate_factor.ox 2>&1 | FileCheck %s --check-prefix=FACTOR
// RUN: not ondrix-compile %S/Inputs/invalid_fir_decimate_shape.ox 2>&1 | FileCheck %s --check-prefix=SHAPE
// RUN: not ondrix-compile %S/Inputs/invalid_fir_decimate_f32.ox 2>&1 | FileCheck %s --check-prefix=F32

// FACTOR: invalid_fir_decimate_factor.ox:3:10: error: fir_decimate source binding currently requires factor=2
// SHAPE: invalid_fir_decimate_shape.ox:3:10: error: static fir_decimate result extent is incorrect

// The resampling kernels are fixed-point contracts; an f32 spelling must be
// rejected at the type check, well before any numeric attribute handling
// could assume a fixed policy.
// F32: invalid_fir_decimate_f32.ox:2:10: error: fir_decimate requires Q15 tensor input, coefficients, and result
