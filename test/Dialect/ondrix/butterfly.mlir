// RUN: ondrix-opt %s | FileCheck %s

// CHECK-LABEL: func.func @butterfly
func.func @butterfly(%a: i32, %b: i32, %tw: i32) -> (i32, i32) {
  // CHECK: ondrix.butterfly
  // CHECK: scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 15, rounding = toward_negative, overflow = saturate, saturate_to = i16>
  %0, %1 = ondrix.butterfly %a, %b, %tw {layout = #ondsp.cx_layout<packed_i16_imag_hi_real_lo>, numeric = #ondsp.fixed<signed, storage = i16, frac = 15>, product = #ondsp.product<full>, scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 15, rounding = toward_negative, overflow = saturate, saturate_to = i16>} : (i32, i32, i32) -> (i32, i32)
  return %0, %1 : i32, i32
}
