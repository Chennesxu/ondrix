// RUN: ondrix-opt %s -split-input-file -verify-diagnostics

// Every operation that can carry an ondsp.cx_layout attribute but has no
// executable packed-Q31 contract rejects the profile explicitly. Only
// ondrix.cfft and ondsp.cx_butterfly implement it in this slice; the rest stay
// Q15-only and must fail closed rather than reinterpret a wider container.

func.func @butterfly_rejects_q31_profile(%a: i64, %b: i64, %twiddle: i64) -> (i64, i64) {
  // expected-error@+1 {{executable butterfly requires packed_i16_imag_hi_real_lo layout}}
  %0, %1 = ondrix.butterfly %a, %b, %twiddle {
    layout = #ondsp.cx_layout<packed_i32_imag_hi_real_lo>,
    numeric = #ondsp.fixed<signed, storage = i32, frac = 31>,
    product = #ondsp.product<full>,
    product_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 31, rounding = nearest_even, overflow = saturate, saturate_to = i32>,
    output_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 1, rounding = nearest_even, overflow = saturate, saturate_to = i32>
  } : (i64, i64, i64) -> (i64, i64)
  return %0, %1 : i64, i64
}

// -----

func.func @rfft_rejects_q31_profile(%input: tensor<8xi32>) -> tensor<5xi64> {
  // expected-error@+1 {{executable RFFT requires packed_i16_imag_hi_real_lo layout}}
  %result = ondrix.rfft %input {
    layout = #ondsp.cx_layout<packed_i32_imag_hi_real_lo>,
    numeric = #ondsp.fixed<signed, storage = i32, frac = 31>,
    product = #ondsp.product<full>,
    product_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 31, rounding = nearest_even, overflow = saturate, saturate_to = i32>,
    output_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 1, rounding = nearest_even, overflow = saturate, saturate_to = i32>
  } : (tensor<8xi32>) -> tensor<5xi64>
  return %result : tensor<5xi64>
}

// -----

func.func @irfft_rejects_q31_profile(%input: tensor<9xi64>) -> tensor<16xi32> {
  // expected-error@+1 {{executable IRFFT requires packed_i16_imag_hi_real_lo layout}}
  %result = ondrix.irfft %input {
    layout = #ondsp.cx_layout<packed_i32_imag_hi_real_lo>,
    numeric = #ondsp.fixed<signed, storage = i32, frac = 31>,
    product = #ondsp.product<full>,
    product_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 31, rounding = nearest_even, overflow = saturate, saturate_to = i32>,
    output_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 1, rounding = nearest_even, overflow = saturate, saturate_to = i32>
  } : (tensor<9xi64>) -> tensor<16xi32>
  return %result : tensor<16xi32>
}

// -----

func.func @rfft_radix4_split_rejects_q31_profile(%input: tensor<32xi16>) -> tensor<17xi32> {
  // expected-error@+1 {{executable radix-4 split RFFT requires packed_i16_imag_hi_real_lo layout}}
  %result = ondrix.rfft_radix4_split %input {
    layout = #ondsp.cx_layout<packed_i32_imag_hi_real_lo>,
    input_numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    output_numeric = #ondsp.fixed<signed, storage = i16, frac = 10>,
    product = #ondsp.product<full>
  } : (tensor<32xi16>) -> tensor<17xi32>
  return %result : tensor<17xi32>
}

// -----

func.func @cx_magnitude_rejects_q31_profile(%input: tensor<5xi64>) -> tensor<5xi16> {
  // expected-error@+1 {{executable magnitude requires packed_i16_imag_hi_real_lo layout}}
  %magnitudes = ondrix.cx_magnitude %input {
    layout = #ondsp.cx_layout<packed_i32_imag_hi_real_lo>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<5xi64>) -> tensor<5xi16>
  return %magnitudes : tensor<5xi16>
}

// -----

// ondsp.cx_mul shares the layout enum but has no packed-Q31 equation. Without
// this branch it would fall through to the unpacked value check and silently
// ignore the declared packing.
func.func @cx_mul_rejects_q31_profile(%lhs: i64, %rhs: i64) -> i64 {
  // expected-error@+1 {{packed_i32_imag_hi_real_lo is supported only by the packed butterfly profile}}
  %0 = ondsp.cx_mul %lhs, %rhs {
    layout = #ondsp.cx_layout<packed_i32_imag_hi_real_lo>,
    numeric = #ondsp.fixed<signed, storage = i32, frac = 31>
  } : (i64, i64) -> i64
  return %0 : i64
}
