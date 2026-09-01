
// RUN: ondrix-compile %S/Inputs/f32_moving_average.ox | FileCheck %s --check-prefix=F32

// F32-LABEL: func.func @f32_moving_average
// F32: ondrix.moving_average
// F32-SAME: numeric = #ondsp.fp<format = f32, contract = off>

// RUN: ondrix-compile %S/Inputs/q31_moving_average.ox | FileCheck %s --check-prefix=Q31

// Q31-LABEL: func.func @q31_moving_average(
// Q31-SAME: %[[X:.*]]: tensor<16xi32>) -> tensor<13xi32>
// Q31: ondrix.moving_average
// Q31-SAME: numeric = #ondsp.fixed<signed, storage = i32, frac = 31>
// Q31-SAME: window = 4
