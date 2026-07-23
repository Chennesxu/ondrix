// RUN: ondrix-opt %s --convert-ondsp-fixed-to-scalar | FileCheck %s

// CHECK-LABEL: func.func @cx_butterfly_q15
// CHECK-NOT: ondsp.
// CHECK-COUNT-4: arith.extsi {{.*}} : i16 to i33
// CHECK-COUNT-4: arith.muli {{.*}} : i33
// CHECK: arith.subi {{.*}} : i33
// CHECK: arith.addi {{.*}} : i33
// CHECK-COUNT-4: arith.extsi {{.*}} : i16 to i17
// CHECK: arith.addi {{.*}} : i17
// CHECK: arith.subi {{.*}} : i17
// CHECK-COUNT-2: arith.ori {{.*}} : i32
func.func @cx_butterfly_q15(%a: i32, %b: i32, %tw: i32) -> (i32, i32) {
  %0, %1 = ondsp.cx_butterfly %a, %b, %tw {
    layout = #ondsp.cx_layout<packed_i16_imag_hi_real_lo>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    product = #ondsp.product<full>,
    product_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 15, rounding = nearest_even, overflow = saturate, saturate_to = i16>,
    output_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 1, rounding = nearest_even, overflow = saturate, saturate_to = i16>
  } : (i32, i32, i32) -> (i32, i32)
  return %0, %1 : i32, i32
}
