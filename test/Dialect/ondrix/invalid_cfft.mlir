// RUN: ondrix-opt %s --split-input-file --verify-diagnostics

func.func @wrong_extent(%input: tensor<16xi32>) -> tensor<16xi32> {
  // expected-error@+1 {{executable CFFT requires matching tensor<4xi32> or tensor<8xi32> input and result}}
  %result = ondrix.cfft %input {
    layout = #ondsp.cx_layout<packed_i16_imag_hi_real_lo>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    product = #ondsp.product<full>,
    product_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 15, rounding = nearest_even, overflow = saturate, saturate_to = i16>,
    output_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 1, rounding = nearest_even, overflow = saturate, saturate_to = i16>
  } : (tensor<16xi32>) -> tensor<16xi32>
  return %result : tensor<16xi32>
}

// -----

func.func @wrong_layout(%input: tensor<4xi32>) -> tensor<4xi32> {
  // expected-error@+1 {{executable CFFT requires packed_i16_imag_hi_real_lo layout}}
  %result = ondrix.cfft %input {
    layout = #ondsp.cx_layout<interleaved>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    product = #ondsp.product<full>,
    product_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 15, rounding = nearest_even, overflow = saturate, saturate_to = i16>,
    output_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 1, rounding = nearest_even, overflow = saturate, saturate_to = i16>
  } : (tensor<4xi32>) -> tensor<4xi32>
  return %result : tensor<4xi32>
}

// -----

func.func @wrong_result(%input: tensor<4xi32>) -> tensor<4xi16> {
  // expected-error@+1 {{executable CFFT requires matching tensor<4xi32> or tensor<8xi32> input and result}}
  %result = ondrix.cfft %input {
    layout = #ondsp.cx_layout<packed_i16_imag_hi_real_lo>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    product = #ondsp.product<full>,
    product_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 15, rounding = nearest_even, overflow = saturate, saturate_to = i16>,
    output_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 1, rounding = nearest_even, overflow = saturate, saturate_to = i16>
  } : (tensor<4xi32>) -> tensor<4xi16>
  return %result : tensor<4xi16>
}
