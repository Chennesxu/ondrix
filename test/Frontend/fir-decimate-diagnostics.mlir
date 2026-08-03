// RUN: not ondrix-compile %S/Inputs/invalid_fir_decimate_factor.ox 2>&1 | FileCheck %s --check-prefix=FACTOR
// RUN: not ondrix-compile %S/Inputs/invalid_fir_decimate_shape.ox 2>&1 | FileCheck %s --check-prefix=SHAPE
// RUN: not ondrix-compile %S/Inputs/invalid_f32_fir_decimate_contract.ox 2>&1 | FileCheck %s --check-prefix=F32

// FACTOR: invalid_fir_decimate_factor.ox:3:10: error: fir_decimate source binding currently requires factor=2
// SHAPE: invalid_fir_decimate_shape.ox:3:10: error: static fir_decimate result extent is incorrect

// An f32 resampling kernel has no default contract to fall back on, so the
// binding requires the declaration rather than choosing one.
// F32: invalid_f32_fir_decimate_contract.ox:2:52: error: expected ',' before fir_decimate contract policy
