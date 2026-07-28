// RUN: not ondrix-opt %s --convert-ondrix-to-ondsp="fft-loops vectorize-static-cfft" 2>&1 | FileCheck %s

// The two alternative FFT lowerings are explicit, mutually exclusive
// profitability choices; selecting both fails closed instead of one
// silently winning.

// CHECK: vectorize-static-cfft and fft-loops are mutually exclusive alternative FFT lowerings; select at most one

func.func @cfft8_forward_q15(%input: tensor<8xi32>) -> tensor<8xi32> {
  %result = ondrix.cfft %input {
    direction = #ondrix.cfft_direction<forward>,
    layout = #ondsp.cx_layout<packed_i16_imag_hi_real_lo>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    product = #ondsp.product<full>,
    product_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 15, rounding = nearest_even, overflow = saturate, saturate_to = i16>,
    output_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 1, rounding = nearest_even, overflow = saturate, saturate_to = i16>
  } : (tensor<8xi32>) -> tensor<8xi32>
  return %result : tensor<8xi32>
}
