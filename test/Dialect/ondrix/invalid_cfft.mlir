// RUN: ondrix-opt %s --split-input-file --verify-diagnostics

func.func @wrong_extent(%input: tensor<6xi32>) -> tensor<6xi32> {
  // expected-error@+1 {{executable CFFT requires matching tensor<Nxi32> input and result with power-of-two N in [4, 1024]}}
  %result = ondrix.cfft %input {
    direction = #ondrix.cfft_direction<forward>,
    layout = #ondsp.cx_layout<packed_i16_imag_hi_real_lo>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    product = #ondsp.product<full>,
    product_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 15, rounding = nearest_even, overflow = saturate, saturate_to = i16>,
    output_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 1, rounding = nearest_even, overflow = saturate, saturate_to = i16>
  } : (tensor<6xi32>) -> tensor<6xi32>
  return %result : tensor<6xi32>
}

// -----

func.func @wrong_layout(%input: tensor<4xi32>) -> tensor<4xi32> {
  // expected-error@+1 {{executable CFFT requires packed_i16_imag_hi_real_lo or packed_i32_imag_hi_real_lo layout}}
  %result = ondrix.cfft %input {
    direction = #ondrix.cfft_direction<forward>,
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
  // expected-error@+1 {{executable CFFT requires matching tensor<Nxi32> input and result with power-of-two N in [4, 1024]}}
  %result = ondrix.cfft %input {
    direction = #ondrix.cfft_direction<forward>,
    layout = #ondsp.cx_layout<packed_i16_imag_hi_real_lo>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    product = #ondsp.product<full>,
    product_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 15, rounding = nearest_even, overflow = saturate, saturate_to = i16>,
    output_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 1, rounding = nearest_even, overflow = saturate, saturate_to = i16>
  } : (tensor<4xi32>) -> tensor<4xi16>
  return %result : tensor<4xi16>
}

// -----

func.func @too_large(%input: tensor<2048xi32>) -> tensor<2048xi32> {
  // expected-error@+1 {{executable CFFT requires matching tensor<Nxi32> input and result with power-of-two N in [4, 1024]}}
  %result = ondrix.cfft %input {
    direction = #ondrix.cfft_direction<forward>,
    layout = #ondsp.cx_layout<packed_i16_imag_hi_real_lo>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    product = #ondsp.product<full>,
    product_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 15, rounding = nearest_even, overflow = saturate, saturate_to = i16>,
    output_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 1, rounding = nearest_even, overflow = saturate, saturate_to = i16>
  } : (tensor<2048xi32>) -> tensor<2048xi32>
  return %result : tensor<2048xi32>
}

// -----

func.func @q31_requires_i64_container(%input: tensor<4xi32>) -> tensor<4xi32> {
  // expected-error@+1 {{executable CFFT requires matching tensor<Nxi64> input and result with power-of-two N in [4, 64]}}
  %result = ondrix.cfft %input {
    direction = #ondrix.cfft_direction<forward>,
    layout = #ondsp.cx_layout<packed_i32_imag_hi_real_lo>,
    numeric = #ondsp.fixed<signed, storage = i32, frac = 31>,
    product = #ondsp.product<high_raw>,
    product_scale = #ondsp.scale<pre_shift_left = 1, post_shift_right = 0, rounding = toward_negative, overflow = saturate, saturate_to = i32>,
    output_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 1, rounding = toward_negative, overflow = saturate, saturate_to = i32>
  } : (tensor<4xi32>) -> tensor<4xi32>
  return %result : tensor<4xi32>
}

// -----

// The Q31 extent bound is the frozen offline twiddle table, not the wider Q15
// bound: an extent the table does not cover must fail closed.
func.func @q31_extent_beyond_frozen_tables(%input: tensor<128xi64>) -> tensor<128xi64> {
  // expected-error@+1 {{executable CFFT requires matching tensor<Nxi64> input and result with power-of-two N in [4, 64]}}
  %result = ondrix.cfft %input {
    direction = #ondrix.cfft_direction<forward>,
    layout = #ondsp.cx_layout<packed_i32_imag_hi_real_lo>,
    numeric = #ondsp.fixed<signed, storage = i32, frac = 31>,
    product = #ondsp.product<high_raw>,
    product_scale = #ondsp.scale<pre_shift_left = 1, post_shift_right = 0, rounding = toward_negative, overflow = saturate, saturate_to = i32>,
    output_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 1, rounding = toward_negative, overflow = saturate, saturate_to = i32>
  } : (tensor<128xi64>) -> tensor<128xi64>
  return %result : tensor<128xi64>
}

// -----

func.func @q31_layout_rejects_q15_numeric(%input: tensor<4xi64>) -> tensor<4xi64> {
  // expected-error@+1 {{packed butterfly requires signed Q31 numeric semantics for this layout}}
  %result = ondrix.cfft %input {
    direction = #ondrix.cfft_direction<forward>,
    layout = #ondsp.cx_layout<packed_i32_imag_hi_real_lo>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    product = #ondsp.product<high_raw>,
    product_scale = #ondsp.scale<pre_shift_left = 1, post_shift_right = 0, rounding = toward_negative, overflow = saturate, saturate_to = i32>,
    output_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 1, rounding = toward_negative, overflow = saturate, saturate_to = i32>
  } : (tensor<4xi64>) -> tensor<4xi64>
  return %result : tensor<4xi64>
}

// -----

func.func @q31_rejects_q15_product_shift(%input: tensor<4xi64>) -> tensor<4xi64> {
  // expected-error@+1 {{product_scale requires pre_shift_left=1 and post_shift_right=0}}
  %result = ondrix.cfft %input {
    direction = #ondrix.cfft_direction<forward>,
    layout = #ondsp.cx_layout<packed_i32_imag_hi_real_lo>,
    numeric = #ondsp.fixed<signed, storage = i32, frac = 31>,
    product = #ondsp.product<high_raw>,
    product_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 15, rounding = toward_negative, overflow = saturate, saturate_to = i32>,
    output_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 1, rounding = toward_negative, overflow = saturate, saturate_to = i32>
  } : (tensor<4xi64>) -> tensor<4xi64>
  return %result : tensor<4xi64>
}

// -----

func.func @q31_rejects_narrow_output_destination(%input: tensor<4xi64>) -> tensor<4xi64> {
  // expected-error@+1 {{output_scale requires signless i32 destination storage}}
  %result = ondrix.cfft %input {
    direction = #ondrix.cfft_direction<forward>,
    layout = #ondsp.cx_layout<packed_i32_imag_hi_real_lo>,
    numeric = #ondsp.fixed<signed, storage = i32, frac = 31>,
    product = #ondsp.product<high_raw>,
    product_scale = #ondsp.scale<pre_shift_left = 1, post_shift_right = 0, rounding = toward_negative, overflow = saturate, saturate_to = i32>,
    output_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 1, rounding = toward_negative, overflow = saturate, saturate_to = i16>
  } : (tensor<4xi64>) -> tensor<4xi64>
  return %result : tensor<4xi64>
}
