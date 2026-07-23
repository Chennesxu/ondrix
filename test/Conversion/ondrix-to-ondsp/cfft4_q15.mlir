// RUN: ondrix-opt %s --convert-ondrix-to-ondsp | FileCheck %s

func.func @cfft4_q15(%input: tensor<4xi32>) -> tensor<4xi32> {
  %result = ondrix.cfft %input {
    layout = #ondsp.cx_layout<packed_i16_imag_hi_real_lo>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    product = #ondsp.product<full>,
    product_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 15, rounding = nearest_even, overflow = saturate, saturate_to = i16>,
    output_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 1, rounding = nearest_even, overflow = saturate, saturate_to = i16>
  } : (tensor<4xi32>) -> tensor<4xi32>
  return %result : tensor<4xi32>
}

// CHECK-LABEL: func.func @cfft4_q15
// CHECK-NOT: ondrix.cfft
// CHECK-DAG: arith.constant 32767 : i32
// CHECK-DAG: arith.constant -2147483648 : i32
// CHECK-COUNT-4: ondsp.cx_butterfly
// CHECK: tensor.insert
