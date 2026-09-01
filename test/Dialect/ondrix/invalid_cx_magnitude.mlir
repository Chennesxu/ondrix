// RUN: ondrix-opt %s --split-input-file --verify-diagnostics

func.func @wrong_layout(%input: tensor<5xi32>) -> tensor<5xi16> {
  // expected-error@+1 {{executable magnitude requires packed_i16_imag_hi_real_lo or packed_i32_imag_hi_real_lo layout}}
  %magnitudes = ondrix.cx_magnitude %input {
    layout = #ondsp.cx_layout<interleaved>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<5xi32>) -> tensor<5xi16>
  return %magnitudes : tensor<5xi16>
}

// -----

func.func @wrong_rounding(%input: tensor<5xi32>) -> tensor<5xi16> {
  // expected-error@+1 {{cx_magnitude supports toward_negative or nearest_even rounding}}
  %magnitudes = ondrix.cx_magnitude %input {
    layout = #ondsp.cx_layout<packed_i16_imag_hi_real_lo>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    rounding = #ondsp.rounding<toward_zero>
  } : (tensor<5xi32>) -> tensor<5xi16>
  return %magnitudes : tensor<5xi16>
}

// -----

func.func @mismatched_extent(%input: tensor<5xi32>) -> tensor<4xi16> {
  // expected-error@+1 {{executable magnitude requires tensor<Nxi32> to tensor<Nxi16> with static N in [1, 4096]}}
  %magnitudes = ondrix.cx_magnitude %input {
    layout = #ondsp.cx_layout<packed_i16_imag_hi_real_lo>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<5xi32>) -> tensor<4xi16>
  return %magnitudes : tensor<4xi16>
}

// -----

func.func @dynamic_extent(%input: tensor<?xi32>) -> tensor<?xi16> {
  // expected-error@+1 {{executable magnitude requires tensor<Nxi32> to tensor<Nxi16> with static N in [1, 4096]}}
  %magnitudes = ondrix.cx_magnitude %input {
    layout = #ondsp.cx_layout<packed_i16_imag_hi_real_lo>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<?xi32>) -> tensor<?xi16>
  return %magnitudes : tensor<?xi16>
}

// -----

func.func @sqrt_wrong_rounding(%input: i64) -> i16 {
  // expected-error@+1 {{sqrt_fixed supports toward_negative or nearest_even rounding}}
  %root = ondsp.sqrt_fixed %input {
    rounding = #ondsp.rounding<toward_zero>
  } : (i64) -> i16
  return %root : i16
}

// -----

func.func @q31_magnitude_without_input_rounding(%input: tensor<5xi64>) -> tensor<5xi32> {
  // expected-error @below {{requantizes each component by 1 before squaring and must declare input_rounding}}
  %magnitudes = ondrix.cx_magnitude %input {
    layout = #ondsp.cx_layout<packed_i32_imag_hi_real_lo>,
    numeric = #ondsp.fixed<signed, storage = i32, frac = 31>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<5xi64>) -> tensor<5xi32>
  return %magnitudes : tensor<5xi32>
}

// -----

func.func @q15_magnitude_with_input_rounding(%input: tensor<5xi32>) -> tensor<5xi16> {
  // expected-error @below {{magnitude at this width has no pre-shift boundary to round}}
  %magnitudes = ondrix.cx_magnitude %input {
    layout = #ondsp.cx_layout<packed_i16_imag_hi_real_lo>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    input_rounding = #ondsp.rounding<nearest_even>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<5xi32>) -> tensor<5xi16>
  return %magnitudes : tensor<5xi16>
}
