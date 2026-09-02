// RUN: ondrix-opt %s --split-input-file --verify-diagnostics

// The split layout names no executable profile; every complex consumer refuses it.
func.func @cfft_split(%input: tensor<4xi32>) -> tensor<4xi32> {
  // expected-error@+1 {{executable CFFT requires packed_i16_imag_hi_real_lo or packed_i32_imag_hi_real_lo layout}}
  %result = ondrix.cfft %input {
    direction = #ondrix.cfft_direction<forward>,
    layout = #ondsp.cx_layout<split>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    product = #ondsp.product<full>,
    product_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 15, rounding = nearest_even, overflow = saturate, saturate_to = i16>,
    output_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 1, rounding = nearest_even, overflow = saturate, saturate_to = i16>
  } : (tensor<4xi32>) -> tensor<4xi32>
  return %result : tensor<4xi32>
}

// -----

func.func @rfft_split(%input: tensor<8xi16>) -> tensor<5xi32> {
  // expected-error@+1 {{executable RFFT requires packed_i16_imag_hi_real_lo or packed_i32_imag_hi_real_lo layout}}
  %result = ondrix.rfft %input {
    layout = #ondsp.cx_layout<split>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    product = #ondsp.product<full>,
    product_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 15, rounding = nearest_even, overflow = saturate, saturate_to = i16>,
    output_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 1, rounding = nearest_even, overflow = saturate, saturate_to = i16>
  } : (tensor<8xi16>) -> tensor<5xi32>
  return %result : tensor<5xi32>
}

// -----

func.func @magnitude_split(%input: tensor<5xi32>) -> tensor<5xi16> {
  // expected-error@+1 {{executable magnitude requires packed_i16_imag_hi_real_lo or packed_i32_imag_hi_real_lo layout}}
  %magnitudes = ondrix.cx_magnitude %input {
    layout = #ondsp.cx_layout<split>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<5xi32>) -> tensor<5xi16>
  return %magnitudes : tensor<5xi16>
}

// -----

func.func @butterfly_split(%a: i32, %b: i32, %tw: i32) -> (i32, i32) {
  // expected-error@+1 {{executable butterfly requires packed_i16_imag_hi_real_lo layout}}
  %0, %1 = ondrix.butterfly %a, %b, %tw {
    layout = #ondsp.cx_layout<split>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    product = #ondsp.product<full>,
    product_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 15, rounding = nearest_even, overflow = saturate, saturate_to = i16>,
    output_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 1, rounding = nearest_even, overflow = saturate, saturate_to = i16>
  } : (i32, i32, i32) -> (i32, i32)
  return %0, %1 : i32, i32
}
