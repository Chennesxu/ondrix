// RUN: ondrix-opt %s --evaluate-ondrix-fir-design | FileCheck %s

// Golden Q15 tables independently derived from the frozen real-valued
// contract with 50-digit mpmath arithmetic; every coefficient clears the
// 2^-20 tie guard by at least 0.0299 LSB.

func.func @window_hamming_even() -> tensor<8xi16> {
  // CHECK-LABEL: func.func @window_hamming_even
  // CHECK-NOT: ondrix.window_hamming
  // CHECK: arith.constant
  // CHECK-SAME: kind = "window_hamming"
  // CHECK-SAME: saturated = 0
  // CHECK-SAME: dense<[2621, 8297, 21049, 31275, 31275, 21049, 8297, 2621]> : tensor<8xi16>
  %window = ondrix.window_hamming {
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>
  } : tensor<8xi16>
  return %window : tensor<8xi16>
}

func.func @window_hamming_odd_saturated_center() -> tensor<9xi16> {
  // CHECK-LABEL: func.func @window_hamming_odd_saturated_center
  // CHECK: arith.constant
  // CHECK-SAME: saturated = 1
  // CHECK-SAME: dense<[2621, 7036, 17695, 28353, 32767, 28353, 17695, 7036, 2621]> : tensor<9xi16>
  %window = ondrix.window_hamming {
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>
  } : tensor<9xi16>
  return %window : tensor<9xi16>
}

func.func @lowpass_quarter() -> tensor<9xi16> {
  // CHECK-LABEL: func.func @lowpass_quarter
  // CHECK-NOT: ondrix.fir_design_windowed_sinc
  // CHECK: arith.constant
  // CHECK-SAME: cutoff_den = 4
  // CHECK-SAME: cutoff_num = 1
  // CHECK-SAME: kind = "fir_design_windowed_sinc"
  // CHECK-SAME: response = "lowpass"
  // CHECK-SAME: saturated = 0
  // CHECK-SAME: dense<[0, -747, 0, 9025, 16384, 9025, 0, -747, 0]> : tensor<9xi16>
  %coefficients = ondrix.fir_design_windowed_sinc {
    response = #ondrix.fir_design_response<lowpass>,
    cutoff_num = 1 : i64, cutoff_den = 4 : i64,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>
  } : tensor<9xi16>
  return %coefficients : tensor<9xi16>
}

func.func @highpass_quarter() -> tensor<9xi16> {
  // CHECK-LABEL: func.func @highpass_quarter
  // CHECK: arith.constant
  // CHECK-SAME: response = "highpass"
  // CHECK-SAME: dense<[0, 747, 0, -9025, 16384, -9025, 0, 747, 0]> : tensor<9xi16>
  %coefficients = ondrix.fir_design_windowed_sinc {
    response = #ondrix.fir_design_response<highpass>,
    cutoff_num = 1 : i64, cutoff_den = 4 : i64,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>
  } : tensor<9xi16>
  return %coefficients : tensor<9xi16>
}

func.func @lowpass_eighth() -> tensor<11xi16> {
  // CHECK-LABEL: func.func @lowpass_eighth
  // CHECK: arith.constant
  // CHECK-SAME: dense<[-118, 0, 978, 3558, 6727, 8192, 6727, 3558, 978, 0, -118]> : tensor<11xi16>
  %coefficients = ondrix.fir_design_windowed_sinc {
    response = #ondrix.fir_design_response<lowpass>,
    cutoff_num = 1 : i64, cutoff_den = 8 : i64,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>
  } : tensor<11xi16>
  return %coefficients : tensor<11xi16>
}
