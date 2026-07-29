// RUN: ondrix-opt %s | FileCheck %s

func.func @window_hamming_even() -> tensor<8xi16> {
  // CHECK: ondrix.window_hamming
  // CHECK-SAME: numeric = #ondsp.fixed<signed, storage = i16, frac = 15>
  // CHECK-SAME: tensor<8xi16>
  %window = ondrix.window_hamming {
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>
  } : tensor<8xi16>
  return %window : tensor<8xi16>
}

func.func @window_hamming_odd() -> tensor<9xi16> {
  // CHECK: ondrix.window_hamming
  // CHECK-SAME: tensor<9xi16>
  %window = ondrix.window_hamming {
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>
  } : tensor<9xi16>
  return %window : tensor<9xi16>
}

func.func @fir_design_lowpass() -> tensor<9xi16> {
  // CHECK: ondrix.fir_design_windowed_sinc
  // CHECK-SAME: cutoff_den = 4
  // CHECK-SAME: cutoff_num = 1
  // CHECK-SAME: response = #ondrix.fir_design_response<lowpass>
  %coefficients = ondrix.fir_design_windowed_sinc {
    response = #ondrix.fir_design_response<lowpass>,
    cutoff_num = 1 : i64, cutoff_den = 4 : i64,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>
  } : tensor<9xi16>
  return %coefficients : tensor<9xi16>
}

func.func @fir_design_highpass() -> tensor<11xi16> {
  // CHECK: ondrix.fir_design_windowed_sinc
  // CHECK-SAME: response = #ondrix.fir_design_response<highpass>
  %coefficients = ondrix.fir_design_windowed_sinc {
    response = #ondrix.fir_design_response<highpass>,
    cutoff_num = 3 : i64, cutoff_den = 16 : i64,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>
  } : tensor<11xi16>
  return %coefficients : tensor<11xi16>
}

// Verifier acceptance witness at both attribute extremes: beta =
// INT64_MAX / INT64_MAX = 1 is inside (0, 50] and exercises the
// representability shortcut (50 * den is not representable, so the bound
// holds without evaluating the product).
func.func @kaiser_maximum_ratio_beta() -> tensor<9xi16> {
  // CHECK: ondrix.window_kaiser
  // CHECK-SAME: beta_den = 9223372036854775807
  // CHECK-SAME: beta_num = 9223372036854775807
  %window = ondrix.window_kaiser {
    beta_num = 9223372036854775807 : i64, beta_den = 9223372036854775807 : i64,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>
  } : tensor<9xi16>
  return %window : tensor<9xi16>
}
