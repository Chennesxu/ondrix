// RUN: ondrix-opt %s --convert-ondrix-to-ondsp="vectorize-static-cfft" -split-input-file -verify-diagnostics

// The Vector-batched CFFT mode still carries hardcoded Q15 lane arithmetic. It
// rejects the packed-Q31 profile instead of emitting an unvalidated schedule.
// The loop-form mode is no longer in that position: it follows the packed
// profile, and its Q31 shape is pinned by cfft_loops_q31.mlir.
func.func @q31_rejects_vector_batched_mode(%input: tensor<8xi64>) -> tensor<8xi64> {
  // expected-error@+2 {{Vector-batched CFFT lowering supports only the packed Q15 profile}}
  // expected-error@+1 {{failed to legalize operation 'ondrix.cfft'}}
  %result = ondrix.cfft %input {
    direction = #ondrix.cfft_direction<forward>,
    layout = #ondsp.cx_layout<packed_i32_imag_hi_real_lo>,
    numeric = #ondsp.fixed<signed, storage = i32, frac = 31>,
    product = #ondsp.product<high_raw>,
    product_scale = #ondsp.scale<pre_shift_left = 1, post_shift_right = 0, rounding = toward_negative, overflow = saturate, saturate_to = i32>,
    output_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 1, rounding = toward_negative, overflow = saturate, saturate_to = i32>
  } : (tensor<8xi64>) -> tensor<8xi64>
  return %result : tensor<8xi64>
}
