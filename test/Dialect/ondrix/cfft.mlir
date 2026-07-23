// RUN: ondrix-opt %s | FileCheck %s

func.func @cfft4_q15(%input: tensor<4xi32>) -> tensor<4xi32> {
  // CHECK: ondrix.cfft
  // CHECK-SAME: layout = #ondsp.cx_layout<packed_i16_imag_hi_real_lo>
  // CHECK-SAME: output_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 1, rounding = nearest_even, overflow = saturate, saturate_to = i16>
  // CHECK-SAME: product = #ondsp.product<full>
  // CHECK-SAME: product_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 15, rounding = nearest_even, overflow = saturate, saturate_to = i16>
  %result = ondrix.cfft %input {
    layout = #ondsp.cx_layout<packed_i16_imag_hi_real_lo>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    product = #ondsp.product<full>,
    product_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 15, rounding = nearest_even, overflow = saturate, saturate_to = i16>,
    output_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 1, rounding = nearest_even, overflow = saturate, saturate_to = i16>
  } : (tensor<4xi32>) -> tensor<4xi32>
  return %result : tensor<4xi32>
}
