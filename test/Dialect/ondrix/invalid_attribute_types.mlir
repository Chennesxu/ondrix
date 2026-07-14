// RUN: ondrix-opt %s -split-input-file -verify-diagnostics

func.func @quantize_rejects_invalid_source_numeric(%input: i32) -> i16 {
  // expected-error@+1 {{attribute 'src' failed to satisfy constraint: ondsp fixed-point or floating-point numeric attribute}}
  %0 = ondrix.quantize %input {src = "invalid", dst = #ondsp.fixed<signed, storage = i16, frac = 15>} : (i32) -> i16
  return %0 : i16
}

// -----

func.func @quantize_rejects_invalid_destination_numeric(%input: i32) -> i16 {
  // expected-error@+1 {{attribute 'dst' failed to satisfy constraint: ondsp fixed-point or floating-point numeric attribute}}
  %0 = ondrix.quantize %input {src = #ondsp.fixed<signed, storage = i32, frac = 30>, dst = "invalid"} : (i32) -> i16
  return %0 : i16
}

// -----

func.func @butterfly_rejects_invalid_layout(
    %a: i32, %b: i32, %twiddle: i32) -> (i32, i32) {
  // expected-error@+1 {{attribute 'layout' failed to satisfy constraint: Complex value storage layout}}
  %0, %1 = ondrix.butterfly %a, %b, %twiddle {layout = "invalid", numeric = #ondsp.fixed<signed, storage = i16, frac = 15>, product = #ondsp.product<full>, scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 15, rounding = trunc, overflow = saturate, saturate_to = i16>} : (i32, i32, i32) -> (i32, i32)
  return %0, %1 : i32, i32
}

// -----

func.func @butterfly_rejects_invalid_numeric(
    %a: i32, %b: i32, %twiddle: i32) -> (i32, i32) {
  // expected-error@+1 {{attribute 'numeric' failed to satisfy constraint: ondsp fixed-point or floating-point numeric attribute}}
  %0, %1 = ondrix.butterfly %a, %b, %twiddle {layout = #ondsp.cx_layout<packed_i16_imag_hi_real_lo>, numeric = "invalid", product = #ondsp.product<full>, scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 15, rounding = trunc, overflow = saturate, saturate_to = i16>} : (i32, i32, i32) -> (i32, i32)
  return %0, %1 : i32, i32
}
