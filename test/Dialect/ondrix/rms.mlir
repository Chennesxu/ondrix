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
