// RUN: ondrix-opt %s --convert-ondrix-to-ondsp | FileCheck %s --check-prefix=STRUCTURE
// RUN: ondrix-opt %s --convert-ondrix-to-ondsp | FileCheck %s --check-prefix=TWIDDLE

func.func @cfft4_q15(%input: tensor<4xi32>) -> tensor<4xi32> {
  %result = ondrix.cfft %input {
    direction = #ondrix.cfft_direction<forward>,
    layout = #ondsp.cx_layout<packed_i16_imag_hi_real_lo>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    product = #ondsp.product<full>,
    product_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 15, rounding = nearest_even, overflow = saturate, saturate_to = i16>,
    output_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 1, rounding = nearest_even, overflow = saturate, saturate_to = i16>
  } : (tensor<4xi32>) -> tensor<4xi32>
  return %result : tensor<4xi32>
}

// STRUCTURE-LABEL: func.func @cfft4_q15
// STRUCTURE-NOT: ondrix.cfft
// STRUCTURE-COUNT-4: ondsp.cx_butterfly
// STRUCTURE: tensor.insert
// TWIDDLE-LABEL: func.func @cfft4_q15
// TWIDDLE-DAG: arith.constant 32767 : i32
// TWIDDLE-DAG: arith.constant -2147483648 : i32

func.func @cfft8_q15(%input: tensor<8xi32>) -> tensor<8xi32> {
  %result = ondrix.cfft %input {
    direction = #ondrix.cfft_direction<forward>,
    layout = #ondsp.cx_layout<packed_i16_imag_hi_real_lo>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    product = #ondsp.product<full>,
    product_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 15, rounding = nearest_even, overflow = saturate, saturate_to = i16>,
    output_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 1, rounding = nearest_even, overflow = saturate, saturate_to = i16>
  } : (tensor<8xi32>) -> tensor<8xi32>
  return %result : tensor<8xi32>
}

// STRUCTURE-LABEL: func.func @cfft8_q15
// STRUCTURE-NOT: ondrix.cfft
// STRUCTURE-COUNT-12: ondsp.cx_butterfly
// STRUCTURE: tensor.insert
// TWIDDLE-LABEL: func.func @cfft8_q15
// TWIDDLE-DAG: arith.constant -1518445950 : i32
// TWIDDLE-DAG: arith.constant -1518426754 : i32

func.func @icfft8_q15(%input: tensor<8xi32>) -> tensor<8xi32> {
  %result = ondrix.cfft %input {
    direction = #ondrix.cfft_direction<inverse>,
    layout = #ondsp.cx_layout<packed_i16_imag_hi_real_lo>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    product = #ondsp.product<full>,
    product_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 15, rounding = nearest_even, overflow = saturate, saturate_to = i16>,
    output_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 1, rounding = nearest_even, overflow = saturate, saturate_to = i16>
  } : (tensor<8xi32>) -> tensor<8xi32>
  return %result : tensor<8xi32>
}

// STRUCTURE-LABEL: func.func @icfft8_q15
// STRUCTURE-NOT: ondrix.cfft
// STRUCTURE-COUNT-12: ondsp.cx_butterfly
// TWIDDLE-LABEL: func.func @icfft8_q15
// TWIDDLE-DAG: arith.constant 1518492290 : i32
// TWIDDLE-DAG: arith.constant 2147418112 : i32
// TWIDDLE-DAG: arith.constant 1518511486 : i32
