// RUN: ondrix-opt %s | FileCheck %s

func.func @rfft8_q15(%input: tensor<8xi16>) -> tensor<5xi32> {
  // CHECK: ondrix.rfft
  // CHECK-SAME: layout = #ondsp.cx_layout<packed_i16_imag_hi_real_lo>
  // CHECK-SAME: numeric = #ondsp.fixed<signed, storage = i16, frac = 15>
  // CHECK-SAME: output_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 1, rounding = nearest_even, overflow = saturate, saturate_to = i16>
  // CHECK-SAME: product = #ondsp.product<full>
  // CHECK-SAME: product_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 15, rounding = nearest_even, overflow = saturate, saturate_to = i16>
  %result = ondrix.rfft %input {
    layout = #ondsp.cx_layout<packed_i16_imag_hi_real_lo>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    product = #ondsp.product<full>,
    product_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 15, rounding = nearest_even, overflow = saturate, saturate_to = i16>,
    output_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 1, rounding = nearest_even, overflow = saturate, saturate_to = i16>
  } : (tensor<8xi16>) -> tensor<5xi32>
  return %result : tensor<5xi32>
}

func.func @irfft16_q15(%input: tensor<9xi32>) -> tensor<16xi16> {
  // CHECK: ondrix.irfft
  %result = ondrix.irfft %input {
    layout = #ondsp.cx_layout<packed_i16_imag_hi_real_lo>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    product = #ondsp.product<full>,
    product_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 15, rounding = nearest_even, overflow = saturate, saturate_to = i16>,
    output_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 1, rounding = nearest_even, overflow = saturate, saturate_to = i16>
  } : (tensor<9xi32>) -> tensor<16xi16>
  return %result : tensor<16xi16>
}
