// RUN: ondrix-opt %s --convert-ondrix-to-ondsp | FileCheck %s

// One saturating floor halving per mirrored pair (three at N=8) is the whole
// toward-negative and clamping surface; the split stage runs no butterfly.

// CHECK-LABEL: func.func @rfft_split8_q31
// CHECK-NOT: ondrix.
// CHECK: ondsp.round_shift
// CHECK-SAME: rounding = toward_zero
// CHECK-COUNT-3: ondsp.sub_shift {{.*}}rounding = toward_negative, overflow = saturate, saturate_to = i32
// CHECK-NOT: ondsp.sub_shift
// CHECK-NOT: ondsp.cx_butterfly
// CHECK-NOT: ondsp.sat_cast
// CHECK: tensor.insert
func.func @rfft_split8_q31(%input: tensor<8xi64>) -> tensor<8xi64> {
  %result = ondrix.rfft_split %input {
    layout = #ondsp.cx_layout<packed_i32_imag_hi_real_lo>,
    input_numeric = #ondsp.fixed<signed, storage = i32, frac = 31>,
    output_numeric = #ondsp.fixed<signed, storage = i32, frac = 31>,
    product = #ondsp.product<full>
  } : (tensor<8xi64>) -> tensor<8xi64>
  return %result : tensor<8xi64>
}
