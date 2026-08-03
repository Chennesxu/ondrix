// RUN: ondrix-opt %s | ondrix-opt | FileCheck %s

// CHECK-LABEL: func.func @gain_q15
// CHECK: ondrix.gain
// CHECK-SAME: gain = 19661
// CHECK-SAME: frac = 15
// CHECK-SAME: rounding = #ondsp.rounding<nearest_even>
func.func @gain_q15(%input: tensor<64xi16>) -> tensor<64xi16> {
  %result = ondrix.gain %input {
    gain = 19661 : i64,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<64xi16>) -> tensor<64xi16>
  return %result : tensor<64xi16>
}

// CHECK-LABEL: func.func @gain_negative_full_scale
// CHECK: ondrix.gain
// CHECK-SAME: gain = -32768
func.func @gain_negative_full_scale(%input: tensor<1xi16>) -> tensor<1xi16> {
  %result = ondrix.gain %input {
    gain = -32768 : i64,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<1xi16>) -> tensor<1xi16>
  return %result : tensor<1xi16>
}

// The second admissible tie rule round-trips as a declared attribute.
// CHECK-LABEL: func.func @gain_ties_positive
// CHECK: ondrix.gain
// CHECK-SAME: gain = 3
// CHECK-SAME: rounding = #ondsp.rounding<nearest_ties_positive>
func.func @gain_ties_positive(%input: tensor<64xi16>) -> tensor<64xi16> {
  %result = ondrix.gain %input {
    gain = 3 : i64,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    rounding = #ondsp.rounding<nearest_ties_positive>
  } : (tensor<64xi16>) -> tensor<64xi16>
  return %result : tensor<64xi16>
}

// CHECK-LABEL: func.func @f32_gain
// CHECK: ondrix.gain
// CHECK-SAME: fp_gain = 2.500000e-01 : f32
// CHECK-SAME: numeric = #ondsp.fp<format = f32, contract = fast>
func.func @f32_gain(%input: tensor<64xf32>) -> tensor<64xf32> {
  %result = ondrix.gain %input {
    fp_gain = 2.500000e-01 : f32,
    numeric = #ondsp.fp<format = f32, contract = fast>
  } : (tensor<64xf32>) -> tensor<64xf32>
  return %result : tensor<64xf32>
}
