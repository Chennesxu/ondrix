// RUN: ondrix-opt %s --split-input-file --verify-diagnostics

func.func @rfft_wrong_extent(%input: tensor<12xi16>) -> tensor<7xi32> {
  // expected-error@+1 {{executable RFFT requires tensor<Nxi16> to tensor<(N/2+1)xi32> with power-of-two N in [8, 1024]}}
  %result = ondrix.rfft %input {
    layout = #ondsp.cx_layout<packed_i16_imag_hi_real_lo>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    product = #ondsp.product<full>,
    product_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 15, rounding = nearest_even, overflow = saturate, saturate_to = i16>,
    output_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 1, rounding = nearest_even, overflow = saturate, saturate_to = i16>
  } : (tensor<12xi16>) -> tensor<7xi32>
  return %result : tensor<7xi32>
}

// -----

func.func @rfft_wrong_layout(%input: tensor<8xi16>) -> tensor<5xi32> {
  // expected-error@+1 {{executable RFFT requires packed_i16_imag_hi_real_lo or packed_i32_imag_hi_real_lo layout}}
  %result = ondrix.rfft %input {
    layout = #ondsp.cx_layout<interleaved>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    product = #ondsp.product<full>,
    product_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 15, rounding = nearest_even, overflow = saturate, saturate_to = i16>,
    output_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 1, rounding = nearest_even, overflow = saturate, saturate_to = i16>
  } : (tensor<8xi16>) -> tensor<5xi32>
  return %result : tensor<5xi32>
}

// -----

func.func @irfft_wrong_shape(%input: tensor<5xi32>) -> tensor<16xi16> {
  // expected-error@+1 {{executable IRFFT requires tensor<(N/2+1)xi32> to tensor<Nxi16> with power-of-two N in [8, 1024]}}
  %result = ondrix.irfft %input {
    layout = #ondsp.cx_layout<packed_i16_imag_hi_real_lo>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    product = #ondsp.product<full>,
    product_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 15, rounding = nearest_even, overflow = saturate, saturate_to = i16>,
    output_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 1, rounding = nearest_even, overflow = saturate, saturate_to = i16>
  } : (tensor<5xi32>) -> tensor<16xi16>
  return %result : tensor<16xi16>
}

// -----

func.func @irfft_wrong_policy(%input: tensor<5xi32>) -> tensor<8xi16> {
  // expected-error@+1 {{output_scale requires pre_shift_left=0 and post_shift_right=1}}
  %result = ondrix.irfft %input {
    layout = #ondsp.cx_layout<packed_i16_imag_hi_real_lo>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    product = #ondsp.product<full>,
    product_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 15, rounding = nearest_even, overflow = saturate, saturate_to = i16>,
    output_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 0, rounding = nearest_even, overflow = saturate, saturate_to = i16>
  } : (tensor<5xi32>) -> tensor<8xi16>
  return %result : tensor<8xi16>
}

// -----

func.func @rfft_encoded_input(
    %input: tensor<8xi16, "encoded">) -> tensor<5xi32> {
  // expected-error@+1 {{does not support encoded tensor types}}
  %result = ondrix.rfft %input {
    layout = #ondsp.cx_layout<packed_i16_imag_hi_real_lo>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    product = #ondsp.product<full>,
    product_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 15, rounding = nearest_even, overflow = saturate, saturate_to = i16>,
    output_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 1, rounding = nearest_even, overflow = saturate, saturate_to = i16>
  } : (tensor<8xi16, "encoded">) -> tensor<5xi32>
  return %result : tensor<5xi32>
}

// -----

func.func @rfft_too_large(%input: tensor<2048xi16>) -> tensor<1025xi32> {
  // expected-error@+1 {{executable RFFT requires tensor<Nxi16> to tensor<(N/2+1)xi32> with power-of-two N in [8, 1024]}}
  %result = ondrix.rfft %input {
    layout = #ondsp.cx_layout<packed_i16_imag_hi_real_lo>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    product = #ondsp.product<full>,
    product_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 15, rounding = nearest_even, overflow = saturate, saturate_to = i16>,
    output_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 1, rounding = nearest_even, overflow = saturate, saturate_to = i16>
  } : (tensor<2048xi16>) -> tensor<1025xi32>
  return %result : tensor<1025xi32>
}

// -----

func.func @fp_wrong_bin_count(%input: tensor<16xf32>) -> tensor<17xf32> {
  // expected-error@+1 {{executable floating-point RFFT requires tensor<Nxf32> to tensor<(N+2)xf32> with power-of-two N in [8, 1024]}}
  %result = ondrix.rfft %input {
    layout = #ondsp.cx_layout<interleaved>,
    numeric = #ondsp.fp<format = f32, contract = off>
  } : (tensor<16xf32>) -> tensor<17xf32>
  return %result : tensor<17xf32>
}

// -----

func.func @fp_declares_a_product(%input: tensor<16xf32>) -> tensor<18xf32> {
  // expected-error@+1 {{floating-point RFFT has no requantization boundary to declare}}
  %result = ondrix.rfft %input {
    layout = #ondsp.cx_layout<interleaved>,
    numeric = #ondsp.fp<format = f32, contract = fma>,
    product = #ondsp.product<full>
  } : (tensor<16xf32>) -> tensor<18xf32>
  return %result : tensor<18xf32>
}

// -----

func.func @fp_irfft_wrong_bin_count(%input: tensor<17xf32>) -> tensor<16xf32> {
  // expected-error@+1 {{executable floating-point IRFFT requires tensor<(N+2)xf32> to tensor<Nxf32> with power-of-two N in [8, 1024]}}
  %result = ondrix.irfft %input {
    layout = #ondsp.cx_layout<interleaved>,
    numeric = #ondsp.fp<format = f32, contract = off>
  } : (tensor<17xf32>) -> tensor<16xf32>
  return %result : tensor<16xf32>
}

// -----

func.func @fp_irfft_declares_a_scale(%input: tensor<18xf32>) -> tensor<16xf32> {
  // expected-error@+1 {{floating-point IRFFT has no requantization boundary to declare}}
  %result = ondrix.irfft %input {
    layout = #ondsp.cx_layout<interleaved>,
    numeric = #ondsp.fp<format = f32, contract = off>,
    product_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 15, rounding = nearest_even, overflow = saturate, saturate_to = i16>
  } : (tensor<18xf32>) -> tensor<16xf32>
  return %result : tensor<16xf32>
}
