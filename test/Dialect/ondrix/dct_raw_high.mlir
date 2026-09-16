// RUN: ondrix-opt %s | ondrix-opt | FileCheck %s

// The raw-high Q31 profile: one floor per term at frac 30, the row sum in the
// shared i40 state, the export shifting by m onto the same reading.
// CHECK-LABEL: func.func @dct8_q31_raw_high
// CHECK: ondrix.dct
// CHECK-SAME: output_numeric = #ondsp.fixed<signed, storage = i32, frac = 27>
// CHECK-SAME: product = #ondsp.product<high_raw>
func.func @dct8_q31_raw_high(%input: tensor<8xi32>) -> tensor<8xi32> {
  %result = ondrix.dct %input {
    input_numeric = #ondsp.fixed<signed, storage = i32, frac = 31>,
    output_numeric = #ondsp.fixed<signed, storage = i32, frac = 27>,
    product = #ondsp.product<high_raw>
  } : (tensor<8xi32>) -> tensor<8xi32>
  return %result : tensor<8xi32>
}
