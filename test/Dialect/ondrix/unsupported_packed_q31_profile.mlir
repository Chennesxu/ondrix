// RUN: ondrix-opt %s -split-input-file -verify-diagnostics

// Every operation that can carry an ondsp.cx_layout attribute but has no
// executable packed-Q31 contract rejects the profile explicitly. ondrix.cfft,
// ondrix.rfft/irfft, and ondsp.cx_butterfly implement it; the rest stay
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

// The Q31 RFFT extent ceiling is the frozen twiddle table, not the Q15 1024.
func.func @rfft_rejects_q31_beyond_table(%input: tensor<128xi32>) -> tensor<65xi64> {
  // expected-error@+1 {{executable RFFT requires tensor<Nxi32> to tensor<(N/2+1)xi64> with power-of-two N in [8, 64]}}
  %result = ondrix.rfft %input {
    layout = #ondsp.cx_layout<packed_i32_imag_hi_real_lo>,
    numeric = #ondsp.fixed<signed, storage = i32, frac = 31>,
    product = #ondsp.product<high_raw>,
    product_scale = #ondsp.scale<pre_shift_left = 1, post_shift_right = 0, rounding = toward_negative, overflow = saturate, saturate_to = i32>,
    output_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 1, rounding = toward_negative, overflow = saturate, saturate_to = i32>
  } : (tensor<128xi32>) -> tensor<65xi64>
  return %result : tensor<65xi64>
}

// -----

// A Q15-shaped value domain under the Q31 layout must not pass either check.
func.func @irfft_rejects_mixed_width_domain(%input: tensor<9xi32>) -> tensor<16xi16> {
  // expected-error@+1 {{executable IRFFT requires tensor<(N/2+1)xi64> to tensor<Nxi32> with power-of-two N in [8, 64]}}
  %result = ondrix.irfft %input {
    layout = #ondsp.cx_layout<packed_i32_imag_hi_real_lo>,
    numeric = #ondsp.fixed<signed, storage = i32, frac = 31>,
    product = #ondsp.product<high_raw>,
    product_scale = #ondsp.scale<pre_shift_left = 1, post_shift_right = 0, rounding = toward_negative, overflow = saturate, saturate_to = i32>,
    output_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 1, rounding = toward_negative, overflow = saturate, saturate_to = i32>
  } : (tensor<9xi32>) -> tensor<16xi16>
  return %result : tensor<16xi16>
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

// -----

// A shape-legal transform whose only defect is the product selection: the
// whole-transform Q31 profile is frozen to raw-high, and full products stay
// admitted only on ondsp.cx_butterfly, where the exact-carrier gate lives.
func.func @cfft_rejects_q31_full_product(%input: tensor<8xi64>) -> tensor<8xi64> {
  // expected-error@+1 {{the packed-Q31 transform profile is frozen to the raw-high product selection}}
  %result = ondrix.cfft %input {
    direction = #ondrix.cfft_direction<forward>,
    layout = #ondsp.cx_layout<packed_i32_imag_hi_real_lo>,
    numeric = #ondsp.fixed<signed, storage = i32, frac = 31>,
    product = #ondsp.product<full>,
    product_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 31, rounding = nearest_even, overflow = saturate, saturate_to = i32>,
    output_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 1, rounding = nearest_even, overflow = saturate, saturate_to = i32>
  } : (tensor<8xi64>) -> tensor<8xi64>
  return %result : tensor<8xi64>
}

// -----

func.func @rfft_rejects_q31_full_product(%input: tensor<8xi32>) -> tensor<5xi64> {
  // expected-error@+1 {{the packed-Q31 transform profile is frozen to the raw-high product selection}}
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

func.func @irfft_rejects_q31_full_product(%input: tensor<5xi64>) -> tensor<8xi32> {
  // expected-error@+1 {{the packed-Q31 transform profile is frozen to the raw-high product selection}}
  %result = ondrix.irfft %input {
    layout = #ondsp.cx_layout<packed_i32_imag_hi_real_lo>,
    numeric = #ondsp.fixed<signed, storage = i32, frac = 31>,
    product = #ondsp.product<full>,
    product_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 31, rounding = nearest_even, overflow = saturate, saturate_to = i32>,
    output_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 1, rounding = nearest_even, overflow = saturate, saturate_to = i32>
  } : (tensor<5xi64>) -> tensor<8xi32>
  return %result : tensor<8xi32>
}
