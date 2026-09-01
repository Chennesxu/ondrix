// RUN: not ondrix-opt %s --split-input-file --evaluate-ondrix-fir-design 2>&1 | FileCheck %s

// The lowpass center tap is exactly 2 * fc = 16385 / 65536, whose Q15
// scaling 8192.5 sits exactly on a rounding tie. The evaluation must fail
// closed instead of trusting binary64 to resolve the real-valued rounding.

// CHECK: error: 'ondrix.fir_design_windowed_sinc' op coefficient 2 lies inside the 2^-20 quantization tie guard; the design profile fails closed

func.func @tie_guarded_lowpass() -> tensor<5xi16> {
  %coefficients = ondrix.fir_design_windowed_sinc {
    response = #ondrix.fir_design_response<lowpass>,
    cutoff_num = 16385 : i64, cutoff_den = 131072 : i64,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>
  } : tensor<5xi16>
  return %coefficients : tensor<5xi16>
}

// -----

// The Q31 window refusal is reachable, not theoretical: 225 is the shortest
// Hamming extent the 2^-13 guard refuses, and every shorter one clears it.

// CHECK: error: 'ondrix.window_hamming' op coefficient 79 lies inside the quantization tie guard for this width; the design profile fails closed

func.func @tie_guarded_hamming_q31() -> tensor<225xi32> {
  %window = ondrix.window_hamming {
    numeric = #ondsp.fixed<signed, storage = i32, frac = 31>
  } : tensor<225xi32>
  return %window : tensor<225xi32>
}
