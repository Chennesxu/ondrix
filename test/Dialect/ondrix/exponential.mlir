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
