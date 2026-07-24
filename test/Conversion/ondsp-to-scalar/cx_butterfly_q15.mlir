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

// CHECK-LABEL: func.func @vector_cx_butterfly_q15
// CHECK-NOT: ondsp.
// CHECK: arith.muli {{.*}} : vector<4xi33>
// CHECK: arith.addi {{.*}} : vector<4xi17>
// CHECK: arith.subi {{.*}} : vector<4xi17>
// CHECK: arith.ori {{.*}} : vector<4xi32>
func.func @vector_cx_butterfly_q15(
    %a: vector<4xi32>, %b: vector<4xi32>, %tw: vector<4xi32>)
    -> (vector<4xi32>, vector<4xi32>) {
  %0, %1 = ondsp.cx_butterfly %a, %b, %tw {
    layout = #ondsp.cx_layout<packed_i16_imag_hi_real_lo>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    product = #ondsp.product<full>,
    product_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 15, rounding = nearest_even, overflow = saturate, saturate_to = i16>,
    output_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 1, rounding = nearest_even, overflow = saturate, saturate_to = i16>
  } : (vector<4xi32>, vector<4xi32>, vector<4xi32>)
      -> (vector<4xi32>, vector<4xi32>)
  return %0, %1 : vector<4xi32>, vector<4xi32>
}
