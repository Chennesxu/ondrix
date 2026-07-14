// RUN: ondrix-opt %s -split-input-file -verify-diagnostics

func.func @sat_cast_rejects_invalid_numeric(%input: i32) -> i16 {
  // expected-error@+1 {{attribute 'numeric' failed to satisfy constraint: ondsp fixed-point or floating-point numeric attribute}}
  %0 = ondsp.sat_cast %input {numeric = "invalid"} : (i32) -> i16
  return %0 : i16
}

// -----

func.func @cx_mul_rejects_invalid_numeric(%lhs: i32, %rhs: i32) -> i32 {
  // expected-error@+1 {{attribute 'numeric' failed to satisfy constraint: ondsp fixed-point or floating-point numeric attribute}}
  %0 = ondsp.cx_mul %lhs, %rhs {layout = #ondsp.cx_layout<packed_i16_imag_hi_real_lo>, numeric = "invalid"} : (i32, i32) -> i32
  return %0 : i32
}

// -----

func.func @fft_stage_rejects_invalid_numeric(%input: i32) -> i32 {
  // expected-error@+1 {{attribute 'numeric' failed to satisfy constraint: ondsp fixed-point or floating-point numeric attribute}}
  %0 = ondsp.fft_stage %input {stage = 0 : i64, layout = #ondsp.cx_layout<packed_i16_imag_hi_real_lo>, numeric = "invalid"} : (i32) -> i32
  return %0 : i32
}
