// RUN: not ondrix-opt %s --convert-ondsp-to-ortumcore 2>&1 | FileCheck %s

func.func @bad_butterfly_storage(%a: i16, %b: i16, %tw: i16) -> (i16, i16) {
  // CHECK: packed i16 butterfly operands must use signless i32 container storage
  %0, %1 = ondsp.cx_butterfly %a, %b, %tw {layout = #ondsp.cx_layout<packed_i16_imag_hi_real_lo>, numeric = #ondsp.fixed<signed, storage = i16, frac = 15>, product = #ondsp.product<full>, scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 15, rounding = toward_negative, overflow = saturate, saturate_to = i16>} : (i16, i16, i16) -> (i16, i16)
  return %0, %1 : i16, i16
}
