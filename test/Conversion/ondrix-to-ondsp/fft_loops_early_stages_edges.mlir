// RUN: ondrix-opt %s --convert-ondrix-to-ondsp="fft-loops fft-loops-vector-width=8" | FileCheck %s --check-prefix=EIGHT
// RUN: ondrix-opt %s --convert-ondrix-to-ondsp="fft-loops fft-loops-vector-width=4" | FileCheck %s --check-prefix=FOUR

// A complete 2*lanes block is required. Unsupported profiles retain their
// old schedule even when they can use Vector arithmetic in later stages.
// EIGHT-LABEL: func.func @small4
// EIGHT-NOT: vector.shuffle
// EIGHT: return
// EIGHT-LABEL: func.func @small8
// EIGHT-NOT: vector.shuffle
// EIGHT: return
// EIGHT-LABEL: func.func @inverse16
// EIGHT: arith.constant dense<[32767, 2147418112, 32767, 2147418112, 32767, 2147418112, 32767, 2147418112]> : vector<8xi32>
// EIGHT: vector.transfer_read {{.*}} : tensor<16xi32>, vector<16xi32>
// EIGHT-COUNT-3: ondsp.cx_butterfly {{.*}} : (vector<8xi32>, vector<8xi32>, vector<8xi32>)
// EIGHT: vector.transfer_write {{.*}} : vector<16xi32>, tensor<16xi32>
// EIGHT-LABEL: func.func @q31
// EIGHT-NOT: vector.shuffle
// EIGHT: ondsp.cx_butterfly {{.*}} : (i64, i64, i64)
// EIGHT-NOT: vector.shuffle
// EIGHT: return
// EIGHT-LABEL: func.func @inventory
// EIGHT-NOT: vector.shuffle
// EIGHT: variant = #ondsp.cx_butterfly_variant<unit>
// EIGHT-NOT: vector.shuffle
// EIGHT: return
// FOUR-LABEL: func.func @small4
// FOUR-NOT: vector.shuffle
// FOUR: return
// FOUR-LABEL: func.func @small8
// FOUR: vector.transfer_read {{.*}} : tensor<8xi32>, vector<8xi32>
// FOUR-COUNT-2: ondsp.cx_butterfly {{.*}} : (vector<4xi32>, vector<4xi32>, vector<4xi32>)
// FOUR: vector.transfer_write {{.*}} : vector<8xi32>, tensor<8xi32>

func.func @small4(%input: tensor<4xi32>) -> tensor<4xi32> {
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

func.func @small8(%input: tensor<8xi32>) -> tensor<8xi32> {
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

func.func @inverse16(%input: tensor<16xi32>) -> tensor<16xi32> {
  %result = ondrix.cfft %input {
    direction = #ondrix.cfft_direction<inverse>,
    layout = #ondsp.cx_layout<packed_i16_imag_hi_real_lo>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    product = #ondsp.product<full>,
    product_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 15, rounding = nearest_even, overflow = saturate, saturate_to = i16>,
    output_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 1, rounding = nearest_even, overflow = saturate, saturate_to = i16>
  } : (tensor<16xi32>) -> tensor<16xi32>
  return %result : tensor<16xi32>
}

func.func @q31(%input: tensor<64xi64>) -> tensor<64xi64> {
  %result = ondrix.cfft %input {
    direction = #ondrix.cfft_direction<forward>,
    layout = #ondsp.cx_layout<packed_i32_imag_hi_real_lo>,
    numeric = #ondsp.fixed<signed, storage = i32, frac = 31>,
    product = #ondsp.product<high_raw>,
    product_scale = #ondsp.scale<pre_shift_left = 1, post_shift_right = 0, rounding = toward_negative, overflow = saturate, saturate_to = i32>,
    output_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 1, rounding = toward_negative, overflow = saturate, saturate_to = i32>
  } : (tensor<64xi64>) -> tensor<64xi64>
  return %result : tensor<64xi64>
}

func.func @inventory(%input: tensor<64xi32>) -> tensor<64xi32> {
  %result = ondrix.cfft %input {
    direction = #ondrix.cfft_direction<forward>,
    layout = #ondsp.cx_layout<packed_i16_imag_hi_real_lo>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    product = #ondsp.product<full>,
    product_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 15, rounding = toward_negative, overflow = saturate, saturate_to = i16>,
    output_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 1, rounding = toward_negative, overflow = saturate, saturate_to = i16>
  } : (tensor<64xi32>) -> tensor<64xi32>
  return %result : tensor<64xi32>
}
