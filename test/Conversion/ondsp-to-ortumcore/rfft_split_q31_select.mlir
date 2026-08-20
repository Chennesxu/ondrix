// RUN: ondrix-opt %s --convert-ondrix-to-ondsp --convert-ondsp-cx-butterfly-to-ortumcore | FileCheck %s

// Mechanically enforces the lowering/selection attribute identity: if the
// split lowering's floor site drifts, the selection stops matching it here.

// CHECK-LABEL: func.func @rfft_split8_q31
// CHECK-COUNT-3: ortumcore.sat_shift_sub %{{.*}} {shift = 1 : i64}
// CHECK-NOT: ondsp.sub_shift
// CHECK-NOT: ortumcore.sat_shift_add
func.func @rfft_split8_q31(%input: tensor<8xi64>) -> tensor<8xi64> {
  %result = ondrix.rfft_split %input {
    layout = #ondsp.cx_layout<packed_i32_imag_hi_real_lo>,
    input_numeric = #ondsp.fixed<signed, storage = i32, frac = 31>,
    output_numeric = #ondsp.fixed<signed, storage = i32, frac = 31>,
    product = #ondsp.product<full>
  } : (tensor<8xi64>) -> tensor<8xi64>
  return %result : tensor<8xi64>
}
