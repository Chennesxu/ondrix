// RUN: ondrix-opt %s --split-input-file --verify-diagnostics

func.func @rfft_split_wrong_layout(%input: tensor<8xi64>) -> tensor<8xi64> {
  // expected-error@+1 {{executable split RFFT requires packed_i32_imag_hi_real_lo layout}}
  %result = ondrix.rfft_split %input {
    layout = #ondsp.cx_layout<packed_i16_imag_hi_real_lo>,
    input_numeric = #ondsp.fixed<signed, storage = i32, frac = 31>,
    output_numeric = #ondsp.fixed<signed, storage = i32, frac = 31>,
    product = #ondsp.product<full>
  } : (tensor<8xi64>) -> tensor<8xi64>
  return %result : tensor<8xi64>
}

// -----

func.func @rfft_split_wrong_product(%input: tensor<8xi64>) -> tensor<8xi64> {
  // expected-error@+1 {{executable split RFFT requires product = #ondsp.product<full>}}
  %result = ondrix.rfft_split %input {
    layout = #ondsp.cx_layout<packed_i32_imag_hi_real_lo>,
    input_numeric = #ondsp.fixed<signed, storage = i32, frac = 31>,
    output_numeric = #ondsp.fixed<signed, storage = i32, frac = 31>,
    product = #ondsp.product<high_raw>
  } : (tensor<8xi64>) -> tensor<8xi64>
  return %result : tensor<8xi64>
}

// -----

func.func @rfft_split_wrong_frac(%input: tensor<8xi64>) -> tensor<8xi64> {
  // expected-error@+1 {{output_numeric requires #ondsp.fixed<signed, storage = i32, frac = 31>}}
  %result = ondrix.rfft_split %input {
    layout = #ondsp.cx_layout<packed_i32_imag_hi_real_lo>,
    input_numeric = #ondsp.fixed<signed, storage = i32, frac = 31>,
    output_numeric = #ondsp.fixed<signed, storage = i32, frac = 30>,
    product = #ondsp.product<full>
  } : (tensor<8xi64>) -> tensor<8xi64>
  return %result : tensor<8xi64>
}

// -----

func.func @rfft_split_extent_two(%input: tensor<2xi64>) -> tensor<2xi64> {
  // expected-error@+1 {{executable split RFFT requires matching tensor<Nxi64> input and result with power-of-two N in [4, 32]}}
  %result = ondrix.rfft_split %input {
    layout = #ondsp.cx_layout<packed_i32_imag_hi_real_lo>,
    input_numeric = #ondsp.fixed<signed, storage = i32, frac = 31>,
    output_numeric = #ondsp.fixed<signed, storage = i32, frac = 31>,
    product = #ondsp.product<full>
  } : (tensor<2xi64>) -> tensor<2xi64>
  return %result : tensor<2xi64>
}

// -----

func.func @rfft_split_extent_sixty_four(%input: tensor<64xi64>) -> tensor<64xi64> {
  // expected-error@+1 {{executable split RFFT requires matching tensor<Nxi64> input and result with power-of-two N in [4, 32]}}
  %result = ondrix.rfft_split %input {
    layout = #ondsp.cx_layout<packed_i32_imag_hi_real_lo>,
    input_numeric = #ondsp.fixed<signed, storage = i32, frac = 31>,
    output_numeric = #ondsp.fixed<signed, storage = i32, frac = 31>,
    product = #ondsp.product<full>
  } : (tensor<64xi64>) -> tensor<64xi64>
  return %result : tensor<64xi64>
}

// -----

func.func @rfft_split_extent_not_power_of_two(%input: tensor<12xi64>) -> tensor<12xi64> {
  // expected-error@+1 {{executable split RFFT requires matching tensor<Nxi64> input and result with power-of-two N in [4, 32]}}
  %result = ondrix.rfft_split %input {
    layout = #ondsp.cx_layout<packed_i32_imag_hi_real_lo>,
    input_numeric = #ondsp.fixed<signed, storage = i32, frac = 31>,
    output_numeric = #ondsp.fixed<signed, storage = i32, frac = 31>,
    product = #ondsp.product<full>
  } : (tensor<12xi64>) -> tensor<12xi64>
  return %result : tensor<12xi64>
}

// -----

func.func @rfft_split_result_extent_mismatch(%input: tensor<8xi64>) -> tensor<5xi64> {
  // expected-error@+1 {{executable split RFFT requires matching tensor<Nxi64> input and result with power-of-two N in [4, 32]}}
  %result = ondrix.rfft_split %input {
    layout = #ondsp.cx_layout<packed_i32_imag_hi_real_lo>,
    input_numeric = #ondsp.fixed<signed, storage = i32, frac = 31>,
    output_numeric = #ondsp.fixed<signed, storage = i32, frac = 31>,
    product = #ondsp.product<full>
  } : (tensor<8xi64>) -> tensor<5xi64>
  return %result : tensor<5xi64>
}

// -----

func.func @rfft_split_wrong_element_type(%input: tensor<8xi32>) -> tensor<8xi32> {
  // expected-error@+1 {{executable split RFFT requires matching tensor<Nxi64> input and result with power-of-two N in [4, 32]}}
  %result = ondrix.rfft_split %input {
    layout = #ondsp.cx_layout<packed_i32_imag_hi_real_lo>,
    input_numeric = #ondsp.fixed<signed, storage = i32, frac = 31>,
    output_numeric = #ondsp.fixed<signed, storage = i32, frac = 31>,
    product = #ondsp.product<full>
  } : (tensor<8xi32>) -> tensor<8xi32>
  return %result : tensor<8xi32>
}
