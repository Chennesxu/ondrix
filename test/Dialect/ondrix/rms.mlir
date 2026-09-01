// RUN: ondrix-opt %s | ondrix-opt | FileCheck %s

// CHECK-LABEL: func.func @rms64_q15
// CHECK: ondrix.rms
// CHECK-SAME: frac = 15
// CHECK-SAME: rounding = #ondsp.rounding<nearest_even>
func.func @rms64_q15(%input: tensor<64xi16>) -> tensor<1xi16> {
  %result = ondrix.rms %input {
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<64xi16>) -> tensor<1xi16>
  return %result : tensor<1xi16>
}

// CHECK-LABEL: func.func @rms_floor_rounding
// CHECK: ondrix.rms
// CHECK-SAME: toward_negative
func.func @rms_floor_rounding(%input: tensor<2xi16>) -> tensor<1xi16> {
  %result = ondrix.rms %input {
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    rounding = #ondsp.rounding<toward_negative>
  } : (tensor<2xi16>) -> tensor<1xi16>
  return %result : tensor<1xi16>
}

// The f32 profile admits any extent in range, because the power-of-two
// requirement exists only to make the fixed-point mean a shift.
// CHECK-LABEL: func.func @f32_rms
// CHECK: ondrix.rms
// CHECK-SAME: numeric = #ondsp.fp<format = f32, contract = off>
// CHECK-NOT: rounding
func.func @f32_rms(%input: tensor<10xf32>) -> tensor<1xf32> {
  %result = ondrix.rms %input {
    numeric = #ondsp.fp<format = f32, contract = off>
  } : (tensor<10xf32>) -> tensor<1xf32>
  return %result : tensor<1xf32>
}

// The Q31 profile carries a third boundary: the extent forces a pre-shift, so
// its rounding is declared alongside the root's.
// CHECK-LABEL: func.func @rms4096_q31
// CHECK: ondrix.rms
// CHECK-SAME: input_rounding = #ondsp.rounding<toward_negative>
// CHECK-SAME: numeric = #ondsp.fixed<signed, storage = i32, frac = 31>
// CHECK-SAME: rounding = #ondsp.rounding<nearest_even>
func.func @rms4096_q31(%input: tensor<4096xi32>) -> tensor<1xi32> {
  %result = ondrix.rms %input {
    numeric = #ondsp.fixed<signed, storage = i32, frac = 31>,
    input_rounding = #ondsp.rounding<toward_negative>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<4096xi32>) -> tensor<1xi32>
  return %result : tensor<1xi32>
}

// The shortest admitted Q31 extent still forces one: two squares of the Q1.31
// minimum already reach 2^63.
// CHECK-LABEL: func.func @rms2_q31
// CHECK: ondrix.rms
// CHECK-SAME: input_rounding
func.func @rms2_q31(%input: tensor<2xi32>) -> tensor<1xi32> {
  %result = ondrix.rms %input {
    numeric = #ondsp.fixed<signed, storage = i32, frac = 31>,
    input_rounding = #ondsp.rounding<nearest_even>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<2xi32>) -> tensor<1xi32>
  return %result : tensor<1xi32>
}
