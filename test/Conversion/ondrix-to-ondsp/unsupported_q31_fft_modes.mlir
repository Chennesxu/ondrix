// RUN: ondrix-opt %s --convert-ondrix-to-ondsp="fft-loops" -split-input-file -verify-diagnostics
// RUN: ondrix-opt %s --convert-ondrix-to-ondsp="vectorize-static-cfft" -split-input-file -verify-diagnostics

// Both opt-in CFFT code-shape modes still carry hardcoded Q15 twiddle tables
// and i32 containers. They reject the packed-Q31 profile instead of emitting
// an unvalidated schedule; only the default unrolled lowering implements it.
func.func @q31_rejects_opt_in_modes(%input: tensor<8xi64>) -> tensor<8xi64> {
  // expected-error@+2 {{CFFT lowering supports only the packed Q15 profile}}
  // expected-error@+1 {{failed to legalize operation 'ondrix.cfft'}}
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
