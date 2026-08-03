
// RUN: ondrix-compile %S/Inputs/f32_moving_average.ox | FileCheck %s --check-prefix=F32

// F32-LABEL: func.func @f32_moving_average
// F32: ondrix.moving_average
// F32-SAME: numeric = #ondsp.fp<format = f32, contract = off>
