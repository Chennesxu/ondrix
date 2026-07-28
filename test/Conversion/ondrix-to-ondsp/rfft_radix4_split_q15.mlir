// RUN: ondrix-opt %s --convert-ondrix-to-ondsp | FileCheck %s

// The half-size radix-4 split schedule routes every rounding decision
// through toward-negative ondsp.round_shift and keeps its only reachable
// saturation as the thirty-two stage-two ondsp.sat_cast clamps. Nothing in
// this contract rounds to nearest.

// CHECK-LABEL: func.func @rfft32_radix4_split_q15
// CHECK-NOT: ondrix.
// CHECK: ondsp.round_shift
// CHECK-SAME: post_shift_right = 2
// CHECK-SAME: rounding = toward_negative
// CHECK-COUNT-32: ondsp.sat_cast {{.*}}#ondsp.fixed<signed, storage = i16, frac = 12>
// CHECK-NOT: ondsp.sat_cast
// CHECK-NOT: nearest_even
// CHECK-NOT: ondsp.cx_butterfly
// CHECK: tensor.insert
func.func @rfft32_radix4_split_q15(%input: tensor<32xi16>) -> tensor<17xi32> {
  %result = ondrix.rfft_radix4_split %input {
    layout = #ondsp.cx_layout<packed_i16_imag_hi_real_lo>,
    input_numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    output_numeric = #ondsp.fixed<signed, storage = i16, frac = 10>,
    product = #ondsp.product<full>
  } : (tensor<32xi16>) -> tensor<17xi32>
  return %result : tensor<17xi32>
}
