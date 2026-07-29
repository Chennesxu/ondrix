// RUN: ondrix-compile %S/Inputs/q15_lms.ox | FileCheck %s
// RUN: not ondrix-compile %S/Inputs/invalid_lms_step_size.ox 2>&1 | FileCheck %s --check-prefix=STEP
// RUN: not ondrix-compile %S/Inputs/invalid_lms_desired.ox 2>&1 | FileCheck %s --check-prefix=DESIRED
// RUN: not ondrix-compile %S/Inputs/invalid_lms_taps.ox 2>&1 | FileCheck %s --check-prefix=TAPS

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
