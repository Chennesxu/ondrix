// RUN: ondrix-opt %s --convert-ondrix-to-ondsp | FileCheck %s

func.func @butterfly_q15(%a: i32, %b: i32, %tw: i32) -> (i32, i32) {
  %0, %1 = ondrix.butterfly %a, %b, %tw {layout = #ondsp.cx_layout<packed_i16_imag_hi_real_lo>, numeric = #ondsp.fixed<signed, storage = i16, frac = 15>, product = #ondsp.product<full>, product_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 15, rounding = nearest_even, overflow = saturate, saturate_to = i16>, output_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 1, rounding = nearest_even, overflow = saturate, saturate_to = i16>, tag = "butterfly"} : (i32, i32, i32) -> (i32, i32)
  return %0, %1 : i32, i32
}

// CHECK-LABEL: func.func @butterfly_q15
// CHECK-NOT: ondrix.butterfly
// CHECK-NOT: tag =
// CHECK: ondsp.cx_butterfly
// CHECK-SAME: #ondsp.cx_layout<packed_i16_imag_hi_real_lo>
// CHECK-SAME: output_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 1, rounding = nearest_even, overflow = saturate, saturate_to = i16>
// CHECK-SAME: #ondsp.product<full>
// CHECK-SAME: product_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 15, rounding = nearest_even, overflow = saturate, saturate_to = i16>
// CHECK-NOT: tag =
