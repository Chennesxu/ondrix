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

func.func @window_hann_odd() -> tensor<9xi16> {
  // CHECK-LABEL: func.func @window_hann_odd
  // CHECK-NOT: ondrix.window_hann
  // CHECK: arith.constant
  // CHECK-SAME: kind = "window_hann"
  // CHECK-SAME: saturated = 1
  // CHECK-SAME: dense<[0, 4799, 16384, 27969, 32767, 27969, 16384, 4799, 0]> : tensor<9xi16>
  %window = ondrix.window_hann {
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>
  } : tensor<9xi16>
  return %window : tensor<9xi16>
}

func.func @window_hann_even() -> tensor<8xi16> {
  // CHECK-LABEL: func.func @window_hann_even
  // CHECK: arith.constant
  // CHECK-SAME: saturated = 0
  // CHECK-SAME: dense<[0, 6169, 20030, 31145, 31145, 20030, 6169, 0]> : tensor<8xi16>
  %window = ondrix.window_hann {
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>
  } : tensor<8xi16>
  return %window : tensor<8xi16>
}

func.func @window_blackman_odd() -> tensor<9xi16> {
  // CHECK-LABEL: func.func @window_blackman_odd
  // CHECK-NOT: ondrix.window_blackman
  // CHECK: arith.constant
  // CHECK-SAME: kind = "window_blackman"
  // CHECK-SAME: saturated = 1
  // CHECK-SAME: dense<[0, 2177, 11141, 25348, 32767, 25348, 11141, 2177, 0]> : tensor<9xi16>
  %window = ondrix.window_blackman {
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>
  } : tensor<9xi16>
  return %window : tensor<9xi16>
}

func.func @window_blackman_even() -> tensor<8xi16> {
  // CHECK-LABEL: func.func @window_blackman_even
  // CHECK: arith.constant
  // CHECK-SAME: saturated = 0
  // CHECK-SAME: dense<[0, 2964, 15047, 30158, 30158, 15047, 2964, 0]> : tensor<8xi16>
  %window = ondrix.window_blackman {
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>
  } : tensor<8xi16>
  return %window : tensor<8xi16>
}

func.func @window_kaiser_odd() -> tensor<9xi16> {
  // CHECK-LABEL: func.func @window_kaiser_odd
  // CHECK-NOT: ondrix.window_kaiser
  // CHECK: arith.constant
  // CHECK-SAME: beta_den = 1
  // CHECK-SAME: beta_num = 6
  // CHECK-SAME: kind = "window_kaiser"
  // CHECK-SAME: saturated = 1
  // CHECK-SAME: dense<[487, 5361, 15825, 27548, 32767, 27548, 15825, 5361, 487]> : tensor<9xi16>
  %window = ondrix.window_kaiser {
    beta_num = 6 : i64, beta_den = 1 : i64,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>
  } : tensor<9xi16>
  return %window : tensor<9xi16>
}

func.func @window_kaiser_even() -> tensor<8xi16> {
  // CHECK-LABEL: func.func @window_kaiser_even
  // CHECK: arith.constant
  // CHECK-SAME: saturated = 0
  // CHECK-SAME: dense<[487, 6546, 19376, 30980, 30980, 19376, 6546, 487]> : tensor<8xi16>
  %window = ondrix.window_kaiser {
    beta_num = 6 : i64, beta_den = 1 : i64,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>
  } : tensor<8xi16>
  return %window : tensor<8xi16>
}

func.func @window_kaiser_huge_denominator() -> tensor<4xi16> {
  // A representable rational far inside (0, 50] whose 50 * beta_den
  // product overflows i64: the bound check must accept it (the review
  // witness for the guarded comparison). At beta = 1e-18 the window is
  // 1.0 everywhere in binary64, so every entry saturates by convention.
  // CHECK-LABEL: func.func @window_kaiser_huge_denominator
  // CHECK: arith.constant
  // CHECK-SAME: saturated = 4
  // CHECK-SAME: dense<32767> : tensor<4xi16>
  %window = ondrix.window_kaiser {
    beta_num = 1 : i64, beta_den = 1000000000000000000 : i64,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>
  } : tensor<4xi16>
  return %window : tensor<4xi16>
}

// The Q31 profile evaluates the same real-valued design at the wider
// quantization. The table below was verified coefficient by coefficient
// against 50-digit mpmath, and it is what the exact sine-argument reduction
// buys: unreduced, sin(pi*x) reaches an argument of pi*2047 at the maximum
// extent and its binary64 error becomes about 3e-03 Q31 LSB, past any guard
// these coefficients could survive.
func.func @lowpass_quarter_q31() -> tensor<9xi32> {
  // CHECK-LABEL: func.func @lowpass_quarter_q31
  // CHECK-NOT: ondrix.fir_design_windowed_sinc
  // CHECK: arith.constant
  // CHECK-SAME: kind = "fir_design_windowed_sinc"
  // CHECK-SAME: saturated = 0
  // CHECK-SAME: dense<[0, -48927525, 0, 591467924, 1073741824, 591467924, 0, -48927525, 0]> : tensor<9xi32>
  %coefficients = ondrix.fir_design_windowed_sinc {
    response = #ondrix.fir_design_response<lowpass>,
    cutoff_num = 1 : i64, cutoff_den = 4 : i64,
    numeric = #ondsp.fixed<signed, storage = i32, frac = 31>
  } : tensor<9xi32>
  return %coefficients : tensor<9xi32>
}

// The four Q31 window tables below were each verified coefficient by
// coefficient against 50-digit mpmath, the Kaiser Bessel series included.

func.func @window_hamming_even_q31() -> tensor<8xi32> {
  // CHECK-LABEL: func.func @window_hamming_even_q31
  // CHECK-NOT: ondrix.window_hamming
  // CHECK: arith.constant
  // CHECK-SAME: kind = "window_hamming"
  // CHECK-SAME: saturated = 0
  // CHECK-SAME: dense<[171798692, 543731459, 1379456801, 2049656489, 2049656489, 1379456801, 543731459, 171798692]> : tensor<8xi32>
  %window = ondrix.window_hamming {
    numeric = #ondsp.fixed<signed, storage = i32, frac = 31>
  } : tensor<8xi32>
  return %window : tensor<8xi32>
}

func.func @window_hann_odd_q31() -> tensor<9xi32> {
  // The exact +1.0 center saturates at the wider width exactly as at Q15,
  // and both endpoints are exactly zero.
  // CHECK-LABEL: func.func @window_hann_odd_q31
  // CHECK: arith.constant
  // CHECK-SAME: kind = "window_hann"
  // CHECK-SAME: saturated = 1
  // CHECK-SAME: dense<[0, 314491699, 1073741824, 1832991949, 2147483647, 1832991949, 1073741824, 314491699, 0]> : tensor<9xi32>
  %window = ondrix.window_hann {
    numeric = #ondsp.fixed<signed, storage = i32, frac = 31>
  } : tensor<9xi32>
  return %window : tensor<9xi32>
}

func.func @window_blackman_even_q31() -> tensor<8xi32> {
  // CHECK-LABEL: func.func @window_blackman_even_q31
  // CHECK: arith.constant
  // CHECK-SAME: kind = "window_blackman"
  // CHECK-SAME: saturated = 0
  // CHECK-SAME: dense<[0, 194247250, 986087893, 1976465820, 1976465820, 986087893, 194247250, 0]> : tensor<8xi32>
  %window = ondrix.window_blackman {
    numeric = #ondsp.fixed<signed, storage = i32, frac = 31>
  } : tensor<8xi32>
  return %window : tensor<8xi32>
}

func.func @window_kaiser_odd_q31() -> tensor<9xi32> {
  // CHECK-LABEL: func.func @window_kaiser_odd_q31
  // CHECK: arith.constant
  // CHECK-SAME: beta_den = 1
  // CHECK-SAME: beta_num = 6
  // CHECK-SAME: kind = "window_kaiser"
  // CHECK-SAME: saturated = 1
  // CHECK-SAME: dense<[31940248, 351344570, 1037139267, 1805356748, 2147483647, 1805356748, 1037139267, 351344570, 31940248]> : tensor<9xi32>
  %window = ondrix.window_kaiser {
    beta_num = 6 : i64, beta_den = 1 : i64,
    numeric = #ondsp.fixed<signed, storage = i32, frac = 31>
  } : tensor<9xi32>
  return %window : tensor<9xi32>
}
