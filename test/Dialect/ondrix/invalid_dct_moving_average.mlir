// RUN: ondrix-opt %s --split-input-file --verify-diagnostics

func.func @dct_bad_extent(%input: tensor<12xi16>) -> tensor<12xi16> {
  // expected-error@+1 {{executable DCT requires matching tensor<Nxi16> input and result with power-of-two N in [4, 64]}}
  %result = ondrix.dct %input {
    input_numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    output_numeric = #ondsp.fixed<signed, storage = i16, frac = 11>
  } : (tensor<12xi16>) -> tensor<12xi16>
  return %result : tensor<12xi16>
}

// -----

func.func @dct_wrong_output_reading(%input: tensor<8xi16>) -> tensor<8xi16> {
  // expected-error@+1 {{output_numeric requires #ondsp.fixed<signed, storage = i16, frac = 11>}}
  %result = ondrix.dct %input {
    input_numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    output_numeric = #ondsp.fixed<signed, storage = i16, frac = 15>
  } : (tensor<8xi16>) -> tensor<8xi16>
  return %result : tensor<8xi16>
}

// -----

func.func @dct_too_large(%input: tensor<128xi16>) -> tensor<128xi16> {
  // expected-error@+1 {{executable DCT requires matching tensor<Nxi16> input and result with power-of-two N in [4, 64]}}
  %result = ondrix.dct %input {
    input_numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    output_numeric = #ondsp.fixed<signed, storage = i16, frac = 7>
  } : (tensor<128xi16>) -> tensor<128xi16>
  return %result : tensor<128xi16>
}

// -----

func.func @average_degenerate_window(%input: tensor<12xi16>) -> tensor<12xi16> {
  // expected-error@+1 {{executable moving average requires a window in [2, 64]}}
  %result = ondrix.moving_average %input {
    window = 1 : i64,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>
  } : (tensor<12xi16>) -> tensor<12xi16>
  return %result : tensor<12xi16>
}

// -----

func.func @average_window_too_large(%input: tensor<70xi16>) -> tensor<6xi16> {
  // expected-error@+1 {{executable moving average requires a window in [2, 64]}}
  %result = ondrix.moving_average %input {
    window = 65 : i64,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>
  } : (tensor<70xi16>) -> tensor<6xi16>
  return %result : tensor<6xi16>
}

// -----

func.func @average_wrong_output(%input: tensor<12xi16>) -> tensor<12xi16> {
  // expected-error@+1 {{executable moving average requires static tensor<Nxi16> input and tensor<(N-K+1)xi16> result with N >= K}}
  %result = ondrix.moving_average %input {
    window = 4 : i64,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>
  } : (tensor<12xi16>) -> tensor<12xi16>
  return %result : tensor<12xi16>
}

// -----

func.func @average_short_input(%input: tensor<4xi16>) -> tensor<1xi16> {
  // expected-error@+1 {{executable moving average requires static tensor<Nxi16> input and tensor<(N-K+1)xi16> result with N >= K}}
  %result = ondrix.moving_average %input {
    window = 8 : i64,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>
  } : (tensor<4xi16>) -> tensor<1xi16>
  return %result : tensor<1xi16>
}

// -----

func.func @fp_dct_output_reading(%input: tensor<8xf32>) -> tensor<8xf32> {
  // expected-error @below {{floating-point DCT output_numeric must equal input_numeric}}
  %result = ondrix.dct %input {
    input_numeric = #ondsp.fp<format = f32, contract = fma>,
    output_numeric = #ondsp.fp<format = f32, contract = off>
  } : (tensor<8xf32>) -> tensor<8xf32>
  return %result : tensor<8xf32>
}
