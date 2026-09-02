// RUN: ondrix-opt %s | ondrix-opt | FileCheck %s

// CHECK-LABEL: func.func @log2_and_exp2
// CHECK: ondrix.log2
// CHECK-SAME: numeric = #ondsp.fixed<unsigned, storage = i16, frac = 16>
// CHECK-SAME: output_numeric = #ondsp.fixed<signed, storage = i16, frac = 11>
// CHECK: ondrix.exp2
// CHECK-SAME: numeric = #ondsp.fixed<signed, storage = i16, frac = 11>
// CHECK-SAME: output_numeric = #ondsp.fixed<unsigned, storage = i16, frac = 16>
func.func @log2_and_exp2(%a: tensor<64xi16>) -> tensor<64xi16> {
  %0 = ondrix.log2 %a {
    numeric = #ondsp.fixed<unsigned, storage = i16, frac = 16>,
    output_numeric = #ondsp.fixed<signed, storage = i16, frac = 11>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<64xi16>) -> tensor<64xi16>
  %1 = ondrix.exp2 %0 {
    numeric = #ondsp.fixed<signed, storage = i16, frac = 11>,
    output_numeric = #ondsp.fixed<unsigned, storage = i16, frac = 16>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<64xi16>) -> tensor<64xi16>
  return %1 : tensor<64xi16>
}

// The Q31 pair: Q0.32 magnitude with Q6.26 exponent, each reading its own.
// CHECK-LABEL: func.func @log2_and_exp2_q31
// CHECK: ondrix.log2
// CHECK-SAME: numeric = #ondsp.fixed<unsigned, storage = i32, frac = 32>
// CHECK-SAME: output_numeric = #ondsp.fixed<signed, storage = i32, frac = 26>
// CHECK: ondrix.exp2
// CHECK-SAME: numeric = #ondsp.fixed<signed, storage = i32, frac = 26>
// CHECK-SAME: output_numeric = #ondsp.fixed<unsigned, storage = i32, frac = 32>
func.func @log2_and_exp2_q31(%a: tensor<64xi32>) -> tensor<64xi32> {
  %0 = ondrix.log2 %a {
    numeric = #ondsp.fixed<unsigned, storage = i32, frac = 32>,
    output_numeric = #ondsp.fixed<signed, storage = i32, frac = 26>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<64xi32>) -> tensor<64xi32>
  %1 = ondrix.exp2 %0 {
    numeric = #ondsp.fixed<signed, storage = i32, frac = 26>,
    output_numeric = #ondsp.fixed<unsigned, storage = i32, frac = 32>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<64xi32>) -> tensor<64xi32>
  return %1 : tensor<64xi32>
}
