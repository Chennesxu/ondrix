// RUN: not ondrix-opt %s --convert-ondsp-to-ortumcore 2>&1 | FileCheck %s

// The abstract target has no proven packed-complex capability at any width, so
// the general lowering refuses the butterfly at both widths. The opt-in
// selection pass reaches the raw-high Q31 profile by DECOMPOSING it into scalar
// capabilities, which is a different statement and a different pass.
func.func @unsupported_cx_butterfly_q31(%a: i64, %b: i64, %tw: i64) -> (i64, i64) {
  // CHECK: failed to legalize operation 'ondsp.cx_butterfly'
  %0, %1 = ondsp.cx_butterfly %a, %b, %tw {
    layout = #ondsp.cx_layout<packed_i32_imag_hi_real_lo>,
    numeric = #ondsp.fixed<signed, storage = i32, frac = 31>,
    product = #ondsp.product<full>,
    product_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 31, rounding = nearest_even, overflow = saturate, saturate_to = i32>,
    output_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 1, rounding = nearest_even, overflow = saturate, saturate_to = i32>
  } : (i64, i64, i64) -> (i64, i64)
  return %0, %1 : i64, i64
}
