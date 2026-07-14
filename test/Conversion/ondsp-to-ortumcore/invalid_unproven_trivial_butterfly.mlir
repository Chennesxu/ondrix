// RUN: not ondrix-opt %s --convert-ondsp-to-ortumcore 2>&1 | FileCheck %s

func.func @unproven_trivial_butterfly(%a: i32, %b: i32, %tw: i32) -> (i32, i32) {
  // CHECK: error: 'ondsp.cx_butterfly' op trivial-twiddle target selection is disabled until the twiddle value, stage role, layout, permutation, and scale are proven
  %0, %1 = ondsp.cx_butterfly %a, %b, %tw {layout = #ondsp.cx_layout<packed_i16_imag_hi_real_lo>, numeric = #ondsp.fixed<signed, storage = i16, frac = 15>, product = #ondsp.product<full>, scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 15, rounding = toward_negative, overflow = saturate, saturate_to = i16>, trivial_twiddle = true} : (i32, i32, i32) -> (i32, i32)
  return %0, %1 : i32, i32
}
