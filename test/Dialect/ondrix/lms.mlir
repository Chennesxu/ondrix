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

// CHECK-LABEL: func.func @f32_lms
// CHECK: ondrix.lms
// CHECK-SAME: fp_step_size = 6.250000e-02 : f32
// CHECK-SAME: numeric = #ondsp.fp<format = f32, contract = fma>
func.func @f32_lms(%input: tensor<32xf32>, %desired: tensor<32xf32>, %weights: tensor<4xf32>)
    -> (tensor<32xf32>, tensor<4xf32>) {
  %error, %adapted = ondrix.lms %input, %desired, %weights {
    fp_step_size = 6.250000e-02 : f32,
    numeric = #ondsp.fp<format = f32, contract = fma>
  } : (tensor<32xf32>, tensor<32xf32>, tensor<4xf32>) -> (tensor<32xf32>, tensor<4xf32>)
  return %error, %adapted : tensor<32xf32>, tensor<4xf32>
}

// The Q31 profile declares a per-product boundary on the tap sum only; the
// step and the update are each a single product and gain nothing. The raw step
// size widens to the Q1.31 range.
// CHECK-LABEL: func.func @lms_k32_q31
// CHECK: ondrix.lms
// CHECK-SAME: numeric = #ondsp.fixed<signed, storage = i32, frac = 31>
// CHECK-SAME: product_rounding = #ondsp.rounding<toward_negative>
// CHECK-SAME: step_size = 268435456
func.func @lms_k32_q31(%x: tensor<16xi32>, %d: tensor<16xi32>, %w: tensor<32xi32>)
    -> (tensor<16xi32>, tensor<32xi32>) {
  %e, %a = ondrix.lms %x, %d, %w {
    numeric = #ondsp.fixed<signed, storage = i32, frac = 31>,
    step_size = 268435456 : i64,
    product_rounding = #ondsp.rounding<toward_negative>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<16xi32>, tensor<16xi32>, tensor<32xi32>) -> (tensor<16xi32>, tensor<32xi32>)
  return %e, %a : tensor<16xi32>, tensor<32xi32>
}

// One tap is one product, which i64 still holds, so this shape declares none.
// CHECK-LABEL: func.func @lms_k1_q31
// CHECK: ondrix.lms
// CHECK-NOT: product_rounding
func.func @lms_k1_q31(%x: tensor<16xi32>, %d: tensor<16xi32>, %w: tensor<1xi32>)
    -> (tensor<16xi32>, tensor<1xi32>) {
  %e, %a = ondrix.lms %x, %d, %w {
    numeric = #ondsp.fixed<signed, storage = i32, frac = 31>,
    step_size = 1024 : i64,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<16xi32>, tensor<16xi32>, tensor<1xi32>) -> (tensor<16xi32>, tensor<1xi32>)
  return %e, %a : tensor<16xi32>, tensor<1xi32>
}
