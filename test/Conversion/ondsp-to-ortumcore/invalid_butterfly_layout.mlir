// RUN: not ondrix-opt %s --convert-ondsp-to-ortumcore 2>&1 | FileCheck %s

func.func @unsupported_packed_butterfly_layout(%a: i32, %b: i32, %tw: i32) -> (i32, i32) {
  // CHECK: ortumcore lowering supports only packed_i16_imag_hi_real_lo layout
  %0, %1 = ondsp.cx_butterfly %a, %b, %tw {layout = #ondsp.cx_layout<packed_i16_real_hi_imag_lo>, numeric = #ondsp.fixed<signed, storage = i16, frac = 15>, product = #ondsp.product<full>, scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 15, rounding = trunc, overflow = saturate, saturate_to = i16>} : (i32, i32, i32) -> (i32, i32)
  return %0, %1 : i32, i32
}
