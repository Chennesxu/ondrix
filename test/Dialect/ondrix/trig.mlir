// RUN: ondrix-opt %s | ondrix-opt | FileCheck %s

// CHECK-LABEL: func.func @sine_q15
// CHECK: ondrix.sine
// CHECK-SAME: frac = 15
// CHECK-SAME: rounding = #ondsp.rounding<nearest_even>
func.func @sine_q15(%phase: tensor<64xi16>) -> tensor<64xi16> {
  %result = ondrix.sine %phase {
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<64xi16>) -> tensor<64xi16>
  return %result : tensor<64xi16>
}

// CHECK-LABEL: func.func @cosine_q15
// CHECK: ondrix.cosine
func.func @cosine_q15(%phase: tensor<1xi16>) -> tensor<1xi16> {
  %result = ondrix.cosine %phase {
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<1xi16>) -> tensor<1xi16>
  return %result : tensor<1xi16>
}
