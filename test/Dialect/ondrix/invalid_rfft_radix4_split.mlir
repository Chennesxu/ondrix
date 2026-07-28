// RUN: ondrix-opt %s --split-input-file --verify-diagnostics

func.func @rfft_radix4_split_wrong_extent(%input: tensor<16xi16>) -> tensor<9xi32> {
  // expected-error@+1 {{executable radix-4 split RFFT requires tensor<32xi16> to tensor<17xi32>}}
  %result = ondrix.rfft_radix4_split %input {
    layout = #ondsp.cx_layout<packed_i16_imag_hi_real_lo>,
    input_numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    output_numeric = #ondsp.fixed<signed, storage = i16, frac = 10>,
    product = #ondsp.product<full>
  } : (tensor<16xi16>) -> tensor<9xi32>
  return %result : tensor<9xi32>
}

// -----

func.func @rfft_radix4_split_wrong_layout(%input: tensor<32xi16>) -> tensor<17xi32> {
  // expected-error@+1 {{executable radix-4 split RFFT requires packed_i16_imag_hi_real_lo layout}}
  %result = ondrix.rfft_radix4_split %input {
    layout = #ondsp.cx_layout<interleaved>,
    input_numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    output_numeric = #ondsp.fixed<signed, storage = i16, frac = 10>,
    product = #ondsp.product<full>
  } : (tensor<32xi16>) -> tensor<17xi32>
  return %result : tensor<17xi32>
}

// -----

func.func @rfft_radix4_split_wrong_input_format(%input: tensor<32xi16>) -> tensor<17xi32> {
  // expected-error@+1 {{input_numeric requires #ondsp.fixed<signed, storage = i16, frac = 15>}}
  %result = ondrix.rfft_radix4_split %input {
    layout = #ondsp.cx_layout<packed_i16_imag_hi_real_lo>,
    input_numeric = #ondsp.fixed<signed, storage = i16, frac = 10>,
    output_numeric = #ondsp.fixed<signed, storage = i16, frac = 10>,
    product = #ondsp.product<full>
  } : (tensor<32xi16>) -> tensor<17xi32>
  return %result : tensor<17xi32>
}

// -----

func.func @rfft_radix4_split_wrong_output_format(%input: tensor<32xi16>) -> tensor<17xi32> {
  // expected-error@+1 {{output_numeric requires #ondsp.fixed<signed, storage = i16, frac = 10>}}
  %result = ondrix.rfft_radix4_split %input {
    layout = #ondsp.cx_layout<packed_i16_imag_hi_real_lo>,
    input_numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    output_numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    product = #ondsp.product<full>
  } : (tensor<32xi16>) -> tensor<17xi32>
  return %result : tensor<17xi32>
}

// -----

func.func @rfft_radix4_split_wrong_product(%input: tensor<32xi16>) -> tensor<17xi32> {
  // expected-error@+1 {{executable radix-4 split RFFT requires product = #ondsp.product<full>}}
  %result = ondrix.rfft_radix4_split %input {
    layout = #ondsp.cx_layout<packed_i16_imag_hi_real_lo>,
    input_numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    output_numeric = #ondsp.fixed<signed, storage = i16, frac = 10>,
    product = #ondsp.product<high_raw>
  } : (tensor<32xi16>) -> tensor<17xi32>
  return %result : tensor<17xi32>
}
