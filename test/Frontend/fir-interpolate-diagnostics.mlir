// RUN: not ondrix-compile %S/Inputs/invalid_fir_interpolate_factor.ox 2>&1 | FileCheck %s --check-prefix=FACTOR
// RUN: not ondrix-compile %S/Inputs/invalid_fir_interpolate_shape.ox 2>&1 | FileCheck %s --check-prefix=SHAPE
// RUN: not ondrix-compile %S/Inputs/invalid_fir_interpolate_overflow.ox 2>&1 | FileCheck %s --check-prefix=OVERFLOW

// FACTOR: invalid_fir_interpolate_factor.ox:3:10: error: fir_interpolate source binding currently requires factor=2
// SHAPE: invalid_fir_interpolate_shape.ox:3:10: error: static fir_interpolate result extent is incorrect
// OVERFLOW: invalid_fir_interpolate_overflow.ox:4:10: error: fir_interpolate result extent overflows index
