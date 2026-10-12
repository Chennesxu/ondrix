// RUN: ondrix-opt %s --convert-ondsp-fixed-to-scalar | FileCheck %s

memref.global "private" constant @safe : memref<8xi32> = dense<[32767, 0xA57E5A82, 0x80000000, 0x80007FFF, 0x7FFF8000, 32767, 0, 32767]>
memref.global "private" constant @unsafe : memref<8xi32> = dense<[32767, 0xA57E5A82, 0x80000000, 0x80007FFF, 0x7FFF8000, 32767, 0, 0x80008000]>
memref.global "private" @mutable : memref<8xi32> = dense<[32767, 0xA57E5A82, 0x80000000, 0x80007FFF, 0x7FFF8000, 32767, 0, 32767]>

// CHECK-LABEL: func.func @tensor_transfer
// CHECK-NOT: i33
// CHECK-COUNT-4: arith.muli {{.*}} : vector<4xi32>
// CHECK-NOT: i33
func.func @tensor_transfer(%a: vector<4xi32>, %b: vector<4xi32>, %i: index) -> (vector<4xi32>, vector<4xi32>) {
  %t = arith.constant dense<[32767, 0xA57E5A82, 0x80000000, 0x80007FFF, 0x7FFF8000, 32767, 0, 32767]> : tensor<8xi32>
  %pad = arith.constant 0 : i32
  %w = vector.transfer_read %t[%i], %pad {in_bounds = [true]} : tensor<8xi32>, vector<4xi32>
  %0, %1 = ondsp.cx_butterfly %a, %b, %w {
    layout = #ondsp.cx_layout<packed_i16_imag_hi_real_lo>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    product = #ondsp.product<full>,
    product_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 15, rounding = nearest_even, overflow = saturate, saturate_to = i16>,
    output_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 1, rounding = nearest_even, overflow = saturate, saturate_to = i16>
  } : (vector<4xi32>, vector<4xi32>, vector<4xi32>) -> (vector<4xi32>, vector<4xi32>)
  return %0, %1 : vector<4xi32>, vector<4xi32>
}

// CHECK-LABEL: func.func @global_transfer
// CHECK-NOT: i33
// CHECK-COUNT-4: arith.muli {{.*}} : vector<4xi32>
// CHECK-NOT: i33
func.func @global_transfer(%a: vector<4xi32>, %b: vector<4xi32>, %i: index) -> (vector<4xi32>, vector<4xi32>) {
  %t = memref.get_global @safe : memref<8xi32>
  %pad = arith.constant 0 : i32
  %w = vector.transfer_read %t[%i], %pad {in_bounds = [true]} : memref<8xi32>, vector<4xi32>
  %0, %1 = ondsp.cx_butterfly %a, %b, %w {
    layout = #ondsp.cx_layout<packed_i16_imag_hi_real_lo>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    product = #ondsp.product<full>,
    product_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 15, rounding = nearest_even, overflow = saturate, saturate_to = i16>,
    output_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 1, rounding = nearest_even, overflow = saturate, saturate_to = i16>
  } : (vector<4xi32>, vector<4xi32>, vector<4xi32>) -> (vector<4xi32>, vector<4xi32>)
  return %0, %1 : vector<4xi32>, vector<4xi32>
}

// CHECK-LABEL: func.func @global_load
// CHECK-COUNT-4: arith.muli {{.*}} : vector<4xi33>
func.func @global_load(%a: vector<4xi32>, %b: vector<4xi32>, %i: index) -> (vector<4xi32>, vector<4xi32>) {
  %t = memref.get_global @safe : memref<8xi32>
  %w = vector.load %t[%i] : memref<8xi32>, vector<4xi32>
  %0, %1 = ondsp.cx_butterfly %a, %b, %w {
    layout = #ondsp.cx_layout<packed_i16_imag_hi_real_lo>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    product = #ondsp.product<full>,
    product_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 15, rounding = nearest_even, overflow = saturate, saturate_to = i16>,
    output_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 1, rounding = nearest_even, overflow = saturate, saturate_to = i16>
  } : (vector<4xi32>, vector<4xi32>, vector<4xi32>) -> (vector<4xi32>, vector<4xi32>)
  return %0, %1 : vector<4xi32>, vector<4xi32>
}

// CHECK-LABEL: func.func @unbounded_table
// CHECK-COUNT-4: arith.muli {{.*}} : vector<4xi33>
func.func @unbounded_table(%a: vector<4xi32>, %b: vector<4xi32>) -> (vector<4xi32>, vector<4xi32>) {
  %t = memref.get_global @unsafe : memref<8xi32>
  %i = arith.constant 4 : index
  %w = vector.load %t[%i] : memref<8xi32>, vector<4xi32>
  %0, %1 = ondsp.cx_butterfly %a, %b, %w {
    layout = #ondsp.cx_layout<packed_i16_imag_hi_real_lo>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    product = #ondsp.product<full>,
    product_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 15, rounding = nearest_even, overflow = saturate, saturate_to = i16>,
    output_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 1, rounding = nearest_even, overflow = saturate, saturate_to = i16>
  } : (vector<4xi32>, vector<4xi32>, vector<4xi32>) -> (vector<4xi32>, vector<4xi32>)
  return %0, %1 : vector<4xi32>, vector<4xi32>
}

// CHECK-LABEL: func.func @mutable_table
// CHECK-COUNT-4: arith.muli {{.*}} : vector<4xi33>
func.func @mutable_table(%a: vector<4xi32>, %b: vector<4xi32>, %i: index) -> (vector<4xi32>, vector<4xi32>) {
  %t = memref.get_global @mutable : memref<8xi32>
  %pad = arith.constant 0 : i32
  %w = vector.transfer_read %t[%i], %pad {in_bounds = [true]} : memref<8xi32>, vector<4xi32>
  %0, %1 = ondsp.cx_butterfly %a, %b, %w {
    layout = #ondsp.cx_layout<packed_i16_imag_hi_real_lo>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    product = #ondsp.product<full>,
    product_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 15, rounding = nearest_even, overflow = saturate, saturate_to = i16>,
    output_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 1, rounding = nearest_even, overflow = saturate, saturate_to = i16>
  } : (vector<4xi32>, vector<4xi32>, vector<4xi32>) -> (vector<4xi32>, vector<4xi32>)
  return %0, %1 : vector<4xi32>, vector<4xi32>
}

// CHECK-LABEL: func.func @runtime_table
// CHECK-COUNT-4: arith.muli {{.*}} : vector<4xi33>
func.func @runtime_table(%a: vector<4xi32>, %b: vector<4xi32>, %i: index, %t: tensor<8xi32>) -> (vector<4xi32>, vector<4xi32>) {
  %pad = arith.constant 0 : i32
  %w = vector.transfer_read %t[%i], %pad {in_bounds = [true]} : tensor<8xi32>, vector<4xi32>
  %0, %1 = ondsp.cx_butterfly %a, %b, %w {
    layout = #ondsp.cx_layout<packed_i16_imag_hi_real_lo>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    product = #ondsp.product<full>,
    product_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 15, rounding = nearest_even, overflow = saturate, saturate_to = i16>,
    output_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 1, rounding = nearest_even, overflow = saturate, saturate_to = i16>
  } : (vector<4xi32>, vector<4xi32>, vector<4xi32>) -> (vector<4xi32>, vector<4xi32>)
  return %0, %1 : vector<4xi32>, vector<4xi32>
}

// CHECK-LABEL: func.func @padded_transfer
// CHECK-COUNT-4: arith.muli {{.*}} : vector<4xi33>
func.func @padded_transfer(%a: vector<4xi32>, %b: vector<4xi32>, %i: index) -> (vector<4xi32>, vector<4xi32>) {
  %t = memref.get_global @safe : memref<8xi32>
  %pad = arith.constant 0x80008000 : i32
  %w = vector.transfer_read %t[%i], %pad : memref<8xi32>, vector<4xi32>
  %0, %1 = ondsp.cx_butterfly %a, %b, %w {
    layout = #ondsp.cx_layout<packed_i16_imag_hi_real_lo>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    product = #ondsp.product<full>,
    product_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 15, rounding = nearest_even, overflow = saturate, saturate_to = i16>,
    output_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 1, rounding = nearest_even, overflow = saturate, saturate_to = i16>
  } : (vector<4xi32>, vector<4xi32>, vector<4xi32>) -> (vector<4xi32>, vector<4xi32>)
  return %0, %1 : vector<4xi32>, vector<4xi32>
}

// CHECK-LABEL: func.func @masked_transfer
// CHECK-COUNT-4: arith.muli {{.*}} : vector<4xi33>
func.func @masked_transfer(%a: vector<4xi32>, %b: vector<4xi32>, %i: index) -> (vector<4xi32>, vector<4xi32>) {
  %t = memref.get_global @safe : memref<8xi32>
  %pad = arith.constant 0x80008000 : i32
  %mask = arith.constant dense<[true, true, true, false]> : vector<4xi1>
  %w = vector.transfer_read %t[%i], %pad, %mask {in_bounds = [true]} : memref<8xi32>, vector<4xi32>
  %0, %1 = ondsp.cx_butterfly %a, %b, %w {
    layout = #ondsp.cx_layout<packed_i16_imag_hi_real_lo>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    product = #ondsp.product<full>,
    product_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 15, rounding = nearest_even, overflow = saturate, saturate_to = i16>,
    output_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 1, rounding = nearest_even, overflow = saturate, saturate_to = i16>
  } : (vector<4xi32>, vector<4xi32>, vector<4xi32>) -> (vector<4xi32>, vector<4xi32>)
  return %0, %1 : vector<4xi32>, vector<4xi32>
}

// CHECK-LABEL: func.func @global_load_exact_end
// CHECK-NOT: i33
// CHECK-COUNT-4: arith.muli {{.*}} : vector<4xi32>
// CHECK-NOT: i33
func.func @global_load_exact_end(%a: vector<4xi32>, %b: vector<4xi32>) -> (vector<4xi32>, vector<4xi32>) {
  %t = memref.get_global @safe : memref<8xi32>
  %i = arith.constant 4 : index
  %w = vector.load %t[%i] : memref<8xi32>, vector<4xi32>
  %0, %1 = ondsp.cx_butterfly %a, %b, %w {
    layout = #ondsp.cx_layout<packed_i16_imag_hi_real_lo>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    product = #ondsp.product<full>,
    product_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 15, rounding = nearest_even, overflow = saturate, saturate_to = i16>,
    output_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 1, rounding = nearest_even, overflow = saturate, saturate_to = i16>
  } : (vector<4xi32>, vector<4xi32>, vector<4xi32>) -> (vector<4xi32>, vector<4xi32>)
  return %0, %1 : vector<4xi32>, vector<4xi32>
}

memref.global "private" constant @short_table : memref<3xi32> = dense<32767>

// CHECK-LABEL: func.func @short_table_crossing
// CHECK-COUNT-4: arith.muli {{.*}} : vector<4xi33>
func.func @short_table_crossing(%a: vector<4xi32>, %b: vector<4xi32>) -> (vector<4xi32>, vector<4xi32>) {
  %t = memref.get_global @short_table : memref<3xi32>
  %i = arith.constant 2 : index
  %w = vector.load %t[%i] : memref<3xi32>, vector<4xi32>
  %0, %1 = ondsp.cx_butterfly %a, %b, %w {
    layout = #ondsp.cx_layout<packed_i16_imag_hi_real_lo>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    product = #ondsp.product<full>,
    product_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 15, rounding = nearest_even, overflow = saturate, saturate_to = i16>,
    output_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 1, rounding = nearest_even, overflow = saturate, saturate_to = i16>
  } : (vector<4xi32>, vector<4xi32>, vector<4xi32>) -> (vector<4xi32>, vector<4xi32>)
  return %0, %1 : vector<4xi32>, vector<4xi32>
}

// CHECK-LABEL: func.func @load_crossing_end
// CHECK-COUNT-4: arith.muli {{.*}} : vector<4xi33>
func.func @load_crossing_end(%a: vector<4xi32>, %b: vector<4xi32>) -> (vector<4xi32>, vector<4xi32>) {
  %t = memref.get_global @safe : memref<8xi32>
  %i = arith.constant 5 : index
  %w = vector.load %t[%i] : memref<8xi32>, vector<4xi32>
  %0, %1 = ondsp.cx_butterfly %a, %b, %w {
    layout = #ondsp.cx_layout<packed_i16_imag_hi_real_lo>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    product = #ondsp.product<full>,
    product_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 15, rounding = nearest_even, overflow = saturate, saturate_to = i16>,
    output_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 1, rounding = nearest_even, overflow = saturate, saturate_to = i16>
  } : (vector<4xi32>, vector<4xi32>, vector<4xi32>) -> (vector<4xi32>, vector<4xi32>)
  return %0, %1 : vector<4xi32>, vector<4xi32>
}

// CHECK-LABEL: func.func @load_negative_index
// CHECK-COUNT-4: arith.muli {{.*}} : vector<4xi33>
func.func @load_negative_index(%a: vector<4xi32>, %b: vector<4xi32>) -> (vector<4xi32>, vector<4xi32>) {
  %t = memref.get_global @safe : memref<8xi32>
  %i = arith.constant -1 : index
  %w = vector.load %t[%i] : memref<8xi32>, vector<4xi32>
  %0, %1 = ondsp.cx_butterfly %a, %b, %w {
    layout = #ondsp.cx_layout<packed_i16_imag_hi_real_lo>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    product = #ondsp.product<full>,
    product_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 15, rounding = nearest_even, overflow = saturate, saturate_to = i16>,
    output_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 1, rounding = nearest_even, overflow = saturate, saturate_to = i16>
  } : (vector<4xi32>, vector<4xi32>, vector<4xi32>) -> (vector<4xi32>, vector<4xi32>)
  return %0, %1 : vector<4xi32>, vector<4xi32>
}

// CHECK-LABEL: func.func @load_large_index
// CHECK-COUNT-4: arith.muli {{.*}} : vector<4xi33>
func.func @load_large_index(%a: vector<4xi32>, %b: vector<4xi32>) -> (vector<4xi32>, vector<4xi32>) {
  %t = memref.get_global @safe : memref<8xi32>
  %i = arith.constant 9223372036854775807 : index
  %w = vector.load %t[%i] : memref<8xi32>, vector<4xi32>
  %0, %1 = ondsp.cx_butterfly %a, %b, %w {
    layout = #ondsp.cx_layout<packed_i16_imag_hi_real_lo>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    product = #ondsp.product<full>,
    product_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 15, rounding = nearest_even, overflow = saturate, saturate_to = i16>,
    output_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 1, rounding = nearest_even, overflow = saturate, saturate_to = i16>
  } : (vector<4xi32>, vector<4xi32>, vector<4xi32>) -> (vector<4xi32>, vector<4xi32>)
  return %0, %1 : vector<4xi32>, vector<4xi32>
}

// CHECK-LABEL: func.func @load_dynamic_extent
// CHECK-COUNT-4: arith.muli {{.*}} : vector<4xi33>
func.func @load_dynamic_extent(%a: vector<4xi32>, %b: vector<4xi32>) -> (vector<4xi32>, vector<4xi32>) {
  %t = memref.get_global @safe : memref<8xi32>
  %view = memref.cast %t : memref<8xi32> to memref<?xi32>
  %i = arith.constant 0 : index
  %w = vector.load %view[%i] : memref<?xi32>, vector<4xi32>
  %0, %1 = ondsp.cx_butterfly %a, %b, %w {
    layout = #ondsp.cx_layout<packed_i16_imag_hi_real_lo>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    product = #ondsp.product<full>,
    product_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 15, rounding = nearest_even, overflow = saturate, saturate_to = i16>,
    output_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 1, rounding = nearest_even, overflow = saturate, saturate_to = i16>
  } : (vector<4xi32>, vector<4xi32>, vector<4xi32>) -> (vector<4xi32>, vector<4xi32>)
  return %0, %1 : vector<4xi32>, vector<4xi32>
}
