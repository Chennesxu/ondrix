// RUN: ondrix-opt %s --split-input-file --verify-diagnostics

func.func @window_wrong_numeric() -> tensor<8xi16> {
  // expected-error@+1 {{numeric requires #ondsp.fixed<signed, storage = i16, frac = 15>}}
  %window = ondrix.window_hamming {
    numeric = #ondsp.fixed<signed, storage = i16, frac = 12>
  } : tensor<8xi16>
  return %window : tensor<8xi16>
}

// -----

func.func @window_too_short() -> tensor<1xi16> {
  // expected-error@+1 {{coefficient extent must be in [2, 4096]}}
  %window = ondrix.window_hamming {
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>
  } : tensor<1xi16>
  return %window : tensor<1xi16>
}

// -----

func.func @window_dynamic() -> tensor<?xi16> {
  // expected-error@+1 {{requires a static rank-1 i16 coefficient tensor}}
  %window = ondrix.window_hamming {
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>
  } : tensor<?xi16>
  return %window : tensor<?xi16>
}

// -----

func.func @design_even_extent() -> tensor<8xi16> {
  // expected-error@+1 {{windowed-sinc design requires an odd coefficient extent}}
  %coefficients = ondrix.fir_design_windowed_sinc {
    response = #ondrix.fir_design_response<lowpass>,
    cutoff_num = 1 : i64, cutoff_den = 4 : i64,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>
  } : tensor<8xi16>
  return %coefficients : tensor<8xi16>
}

// -----

func.func @design_zero_cutoff() -> tensor<9xi16> {
  // expected-error@+1 {{cutoff requires 1 <= cutoff_num and 2 * cutoff_num < cutoff_den}}
  %coefficients = ondrix.fir_design_windowed_sinc {
    response = #ondrix.fir_design_response<lowpass>,
    cutoff_num = 0 : i64, cutoff_den = 4 : i64,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>
  } : tensor<9xi16>
  return %coefficients : tensor<9xi16>
}

// -----

func.func @design_nyquist_cutoff() -> tensor<9xi16> {
  // expected-error@+1 {{cutoff requires 1 <= cutoff_num and 2 * cutoff_num < cutoff_den}}
  %coefficients = ondrix.fir_design_windowed_sinc {
    response = #ondrix.fir_design_response<highpass>,
    cutoff_num = 1 : i64, cutoff_den = 2 : i64,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>
  } : tensor<9xi16>
  return %coefficients : tensor<9xi16>
}

// -----

func.func @design_wrong_element_type() -> tensor<9xi32> {
  // expected-error@+1 {{requires a static rank-1 i16 coefficient tensor}}
  %coefficients = ondrix.fir_design_windowed_sinc {
    response = #ondrix.fir_design_response<lowpass>,
    cutoff_num = 1 : i64, cutoff_den = 4 : i64,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>
  } : tensor<9xi32>
  return %coefficients : tensor<9xi32>
}

// -----

func.func @design_too_long() -> tensor<4097xi16> {
  // expected-error@+1 {{coefficient extent must be in [3, 4095]}}
  %coefficients = ondrix.fir_design_windowed_sinc {
    response = #ondrix.fir_design_response<lowpass>,
    cutoff_num = 1 : i64, cutoff_den = 4 : i64,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>
  } : tensor<4097xi16>
  return %coefficients : tensor<4097xi16>
}

// -----

func.func @kaiser_zero_beta() -> tensor<9xi16> {
  // expected-error@+1 {{kaiser beta must be a positive rational in (0, 50]}}
  %window = ondrix.window_kaiser {
    beta_num = 0 : i64, beta_den = 1 : i64,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>
  } : tensor<9xi16>
  return %window : tensor<9xi16>
}

// -----

func.func @kaiser_oversized_beta() -> tensor<9xi16> {
  // expected-error@+1 {{kaiser beta must be a positive rational in (0, 50]}}
  %window = ondrix.window_kaiser {
    beta_num = 101 : i64, beta_den = 2 : i64,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>
  } : tensor<9xi16>
  return %window : tensor<9xi16>
}
