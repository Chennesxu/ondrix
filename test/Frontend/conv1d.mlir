// RUN: ondrix-compile %S/Inputs/q15_convolution.ox | FileCheck %s --check-prefix=Q15
// RUN: ondrix-compile %S/Inputs/q31_correlation.ox | FileCheck %s --check-prefix=Q31
// RUN: ondrix-compile %S/Inputs/f32_correlation.ox | FileCheck %s --check-prefix=F32
// RUN: not ondrix-compile %S/Inputs/invalid_convolution_shape.ox 2>&1 | FileCheck %s --check-prefix=SHAPE

// Q15-LABEL: func.func @q15_convolution(
// Q15-SAME: tensor<6xi16>
// Q15: %[[RESULT:.*]] = ondrix.conv1d
// Q15-SAME: mode = #ondrix.conv1d_mode<convolution>
// Q15-SAME: numeric = #ondsp.fixed<signed, storage = i16, frac = 15>
// Q15: return %[[RESULT]] : tensor<4xi16>

// Q31-LABEL: func.func @q31_correlation(
// Q31-SAME: tensor<6xi32>
// Q31: %[[RESULT:.*]] = ondrix.conv1d
// Q31-SAME: accumulator = !ondsp.acc<storage = i64, frac = 62, signed, update_overflow = wrap>
// Q31-SAME: mode = #ondrix.conv1d_mode<correlation>
// Q31-SAME: numeric = #ondsp.fixed<signed, storage = i32, frac = 31>
// Q31: return %[[RESULT]] : tensor<4xi32>

// F32-LABEL: func.func @f32_correlation(
// F32-SAME: tensor<6xf32>
// F32: %[[RESULT:.*]] = ondrix.conv1d
// F32-SAME: mode = #ondrix.conv1d_mode<correlation>
// F32-SAME: numeric = #ondsp.fp<format = f32, contract = fma>
// F32: return %[[RESULT]] : tensor<4xf32>

// SHAPE: invalid_convolution_shape.ox:2:10: error: static convolution/correlation result extent is incorrect
