// RUN: ondrix-compile %S/Inputs/q15_butterfly.ox | FileCheck %s

// CHECK-LABEL: func.func @q15_butterfly(
// CHECK-SAME: %[[A:.*]]: i32, %[[B:.*]]: i32, %[[W:.*]]: i32) -> (i32, i32)
// CHECK: %[[OUT0:.*]], %[[OUT1:.*]] = ondrix.butterfly %[[A]], %[[B]], %[[W]]
// CHECK-SAME: layout = #ondsp.cx_layout<packed_i16_imag_hi_real_lo>
// CHECK-SAME: numeric = #ondsp.fixed<signed, storage = i16, frac = 15>
// CHECK-SAME: output_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 1, rounding = nearest_even, overflow = saturate, saturate_to = i16>
// CHECK-SAME: product = #ondsp.product<full>
// CHECK-SAME: product_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 15, rounding = nearest_even, overflow = saturate, saturate_to = i16>
// CHECK: return %[[OUT0]], %[[OUT1]] : i32, i32
