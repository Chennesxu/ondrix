// RUN: ondrix-opt %s | ondrix-opt | FileCheck %s

// CHECK-LABEL: func.func @lms8_q15
// CHECK: ondrix.lms
// CHECK-SAME: frac = 15
// CHECK-SAME: rounding = #ondsp.rounding<nearest_even>
// CHECK-SAME: step_size = 4096
func.func @lms8_q15(%x: tensor<256xi16>, %d: tensor<256xi16>, %w: tensor<8xi16>)
    -> (tensor<256xi16>, tensor<8xi16>) {
  %e, %wf = ondrix.lms %x, %d, %w {
    step_size = 4096 : i64,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<256xi16>, tensor<256xi16>, tensor<8xi16>) -> (tensor<256xi16>, tensor<8xi16>)
  return %e, %wf : tensor<256xi16>, tensor<8xi16>
}

// CHECK-LABEL: func.func @lms_single_tap
// CHECK: ondrix.lms
func.func @lms_single_tap(%x: tensor<1xi16>, %d: tensor<1xi16>, %w: tensor<1xi16>)
    -> (tensor<1xi16>, tensor<1xi16>) {
  %e, %wf = ondrix.lms %x, %d, %w {
    step_size = 0 : i64,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<1xi16>, tensor<1xi16>, tensor<1xi16>) -> (tensor<1xi16>, tensor<1xi16>)
  return %e, %wf : tensor<1xi16>, tensor<1xi16>
}
