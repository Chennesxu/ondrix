// RUN: ondrix-opt %s --convert-ondsp-fixed-to-scalar | FileCheck %s

// CHECK-LABEL: func.func @bounded_lanes
// CHECK-NOT: i33
// CHECK-COUNT-4: arith.muli {{.*}} : vector<4xi32>
// CHECK: arith.subi {{.*}} : vector<4xi32>
// CHECK: arith.addi {{.*}} : vector<4xi32>
// CHECK-NOT: i33
func.func @bounded_lanes(%a: vector<4xi32>, %b: vector<4xi32>) -> (vector<4xi32>, vector<4xi32>) {
  %tw = arith.constant dense<[-1518445950, -2147450881, 2147450880, -2147483648]> : vector<4xi32>
  %0, %1 = ondsp.cx_butterfly %a, %b, %tw {
    layout = #ondsp.cx_layout<packed_i16_imag_hi_real_lo>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    product = #ondsp.product<full>,
    product_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 15, rounding = nearest_even, overflow = saturate, saturate_to = i16>,
    output_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 1, rounding = nearest_even, overflow = saturate, saturate_to = i16>
  } : (vector<4xi32>, vector<4xi32>, vector<4xi32>) -> (vector<4xi32>, vector<4xi32>)
  return %0, %1 : vector<4xi32>, vector<4xi32>
}

// CHECK-LABEL: func.func @bounded_splat
// CHECK-NOT: i33
// CHECK-COUNT-4: arith.muli {{.*}} : vector<4xi32>
// CHECK-NOT: i33
func.func @bounded_splat(%a: vector<4xi32>, %b: vector<4xi32>) -> (vector<4xi32>, vector<4xi32>) {
  %tw = arith.constant dense<-2147450881> : vector<4xi32>
  %0, %1 = ondsp.cx_butterfly %a, %b, %tw {
    layout = #ondsp.cx_layout<packed_i16_imag_hi_real_lo>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    product = #ondsp.product<full>,
    product_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 15, rounding = nearest_even, overflow = saturate, saturate_to = i16>,
    output_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 1, rounding = nearest_even, overflow = saturate, saturate_to = i16>
  } : (vector<4xi32>, vector<4xi32>, vector<4xi32>) -> (vector<4xi32>, vector<4xi32>)
  return %0, %1 : vector<4xi32>, vector<4xi32>
}

// The last lane alone reaches 2^31, withdrawing narrowing for the entire vector.
// CHECK-LABEL: func.func @unbounded_last_lane
// CHECK-COUNT-4: arith.muli {{.*}} : vector<4xi33>
// CHECK: arith.addi {{.*}} : vector<4xi33>
func.func @unbounded_last_lane(%a: vector<4xi32>, %b: vector<4xi32>) -> (vector<4xi32>, vector<4xi32>) {
  %tw = arith.constant dense<[-1518445950, -2147450881, 2147450880, -2147450880]> : vector<4xi32>
  %0, %1 = ondsp.cx_butterfly %a, %b, %tw {
    layout = #ondsp.cx_layout<packed_i16_imag_hi_real_lo>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    product = #ondsp.product<full>,
    product_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 15, rounding = nearest_even, overflow = saturate, saturate_to = i16>,
    output_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 1, rounding = nearest_even, overflow = saturate, saturate_to = i16>
  } : (vector<4xi32>, vector<4xi32>, vector<4xi32>) -> (vector<4xi32>, vector<4xi32>)
  return %0, %1 : vector<4xi32>, vector<4xi32>
}
