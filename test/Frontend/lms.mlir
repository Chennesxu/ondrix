// RUN: ondrix-compile %S/Inputs/q15_lms.ox | FileCheck %s
// RUN: ondrix-compile %S/Inputs/q31_lms.ox | FileCheck %s --check-prefix=Q31
// RUN: ondrix-compile %S/Inputs/q31_lms_floor.ox | FileCheck %s --check-prefix=Q31FLOOR
// RUN: not ondrix-compile %S/Inputs/invalid_lms_product_rounding.ox 2>&1 | FileCheck %s --check-prefix=TAPBOUND
// RUN: not ondrix-compile %S/Inputs/invalid_lms_step_size.ox 2>&1 | FileCheck %s --check-prefix=STEP
// RUN: not ondrix-compile %S/Inputs/invalid_lms_desired.ox 2>&1 | FileCheck %s --check-prefix=DESIRED
// RUN: not ondrix-compile %S/Inputs/invalid_lms_taps.ox 2>&1 | FileCheck %s --check-prefix=TAPS
// RUN: ondrix-compile %S/Inputs/f32_lms.ox | FileCheck %s --check-prefix=FP

// CHECK-LABEL: func.func @q15_lms(
// CHECK-SAME: tensor<64xi16>
// CHECK-SAME: tensor<64xi16>
// CHECK-SAME: tensor<8xi16>
// CHECK-SAME: -> (tensor<64xi16>, tensor<8xi16>)
// CHECK: %[[ERROR:.*]], %[[ADAPTED:.*]] = ondrix.lms
// CHECK-SAME: numeric = #ondsp.fixed<signed, storage = i16, frac = 15>
// CHECK-SAME: rounding = #ondsp.rounding<nearest_even>
// CHECK-SAME: step_size = 4096 : i64
// CHECK-SAME: -> (tensor<64xi16>, tensor<8xi16>)
// CHECK: return %[[ERROR]], %[[ADAPTED]]

// STEP: invalid_lms_step_size.ox:2:10: error: lms step size must be a raw signed Q1.15 value in [0, 32767]
// DESIRED: invalid_lms_desired.ox:2:10: error: lms input, desired, and error extents must match
// TAPS: invalid_lms_taps.ox:2:10: error: lms currently requires a weight extent in [1, 64]

// FP-LABEL: func.func @f32_lms(
// FP: ondrix.lms
// FP-SAME: fp_step_size = 6.250000e-02 : f32
// FP-SAME: numeric = #ondsp.fp<format = f32, contract = fma>

// The binding supplies the tap-sum boundary the tap count forces, and the raw
// step size widens to the Q1.31 range.
// Q31-LABEL: func.func @q31_lms(
// Q31: ondrix.lms
// Q31-SAME: numeric = #ondsp.fixed<signed, storage = i32, frac = 31>
// Q31-SAME: product_rounding = #ondsp.rounding<nearest_even>
// Q31-SAME: step_size = 268435456

// Q31FLOOR-LABEL: func.func @q31_lms_floor(
// Q31FLOOR: ondrix.lms
// Q31FLOOR-SAME: product_rounding = #ondsp.rounding<toward_negative>

// TAPBOUND: lms at this width and tap count has no product boundary to round
