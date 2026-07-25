// RUN: not ondrix-compile %S/Inputs/invalid_fir_decimate_factor.ox 2>&1 | FileCheck %s --check-prefix=FACTOR
// RUN: not ondrix-compile %S/Inputs/invalid_fir_decimate_shape.ox 2>&1 | FileCheck %s --check-prefix=SHAPE

// FACTOR: invalid_fir_decimate_factor.ox:3:10: error: fir_decimate source binding currently requires factor=2
// SHAPE: invalid_fir_decimate_shape.ox:3:10: error: static fir_decimate result extent is incorrect
