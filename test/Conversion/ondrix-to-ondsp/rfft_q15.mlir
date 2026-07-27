// RUN: ondrix-opt %s --convert-ondrix-to-ondsp | FileCheck %s

func.func @rfft8_q15(%input: tensor<8xi16>) -> tensor<5xi32> {
  %result = ondrix.rfft %input {
    layout = #ondsp.cx_layout<packed_i16_imag_hi_real_lo>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    product = #ondsp.product<full>,
    product_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 15, rounding = nearest_even, overflow = saturate, saturate_to = i16>,
    output_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 1, rounding = nearest_even, overflow = saturate, saturate_to = i16>
  } : (tensor<8xi16>) -> tensor<5xi32>
  return %result : tensor<5xi32>
}

// CHECK-LABEL: func.func @rfft8_q15
// CHECK-NOT: ondrix.rfft
// CHECK-COUNT-12: ondsp.cx_butterfly
// CHECK: arith.trunci {{.*}} : i32 to i16
// CHECK-NEXT: arith.extui {{.*}} : i16 to i32
// CHECK: tensor.insert {{.*}} : tensor<5xi32>

func.func @rfft16_q15(%input: tensor<16xi16>) -> tensor<9xi32> {
  %result = ondrix.rfft %input {
    layout = #ondsp.cx_layout<packed_i16_imag_hi_real_lo>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    product = #ondsp.product<full>,
    product_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 15, rounding = nearest_even, overflow = saturate, saturate_to = i16>,
    output_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 1, rounding = nearest_even, overflow = saturate, saturate_to = i16>
  } : (tensor<16xi16>) -> tensor<9xi32>
  return %result : tensor<9xi32>
}

// CHECK-LABEL: func.func @rfft16_q15
// CHECK-NOT: ondrix.rfft
// CHECK-COUNT-32: ondsp.cx_butterfly
// CHECK: tensor.insert {{.*}} : tensor<9xi32>

func.func @irfft8_q15(%input: tensor<5xi32>) -> tensor<8xi16> {
  %result = ondrix.irfft %input {
    layout = #ondsp.cx_layout<packed_i16_imag_hi_real_lo>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    product = #ondsp.product<full>,
    product_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 15, rounding = nearest_even, overflow = saturate, saturate_to = i16>,
    output_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 1, rounding = nearest_even, overflow = saturate, saturate_to = i16>
  } : (tensor<5xi32>) -> tensor<8xi16>
  return %result : tensor<8xi16>
}

// CHECK-LABEL: func.func @irfft8_q15
// CHECK-NOT: ondrix.irfft
// CHECK: arith.cmpi eq, {{.*}}, %c-32768_i16
// CHECK: arith.select
// CHECK-COUNT-12: ondsp.cx_butterfly
// CHECK: tensor.insert {{.*}} : tensor<8xi16>

func.func @irfft16_q15(%input: tensor<9xi32>) -> tensor<16xi16> {
  %result = ondrix.irfft %input {
    layout = #ondsp.cx_layout<packed_i16_imag_hi_real_lo>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    product = #ondsp.product<full>,
    product_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 15, rounding = nearest_even, overflow = saturate, saturate_to = i16>,
    output_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 1, rounding = nearest_even, overflow = saturate, saturate_to = i16>
  } : (tensor<9xi32>) -> tensor<16xi16>
  return %result : tensor<16xi16>
}

// CHECK-LABEL: func.func @irfft16_q15
// CHECK-NOT: ondrix.irfft
// CHECK-COUNT-32: ondsp.cx_butterfly
// CHECK: tensor.insert {{.*}} : tensor<16xi16>
