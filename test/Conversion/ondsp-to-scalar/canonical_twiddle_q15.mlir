// RUN: ondrix-opt %s --convert-ondsp-fixed-to-scalar="specialize-canonical-twiddles" | FileCheck %s

// CHECK-LABEL: func.func @canonical_one
// CHECK-NOT: arith.muli
// CHECK: arith.cmpi slt
// CHECK: arith.cmpi sgt
// CHECK-LABEL: func.func @canonical_minus_j
// CHECK-NOT: arith.muli
// CHECK: arith.cmpi eq
// CHECK-LABEL: func.func @noncanonical_constant
// CHECK-COUNT-4: arith.muli
// CHECK-LABEL: func.func @runtime_twiddle
// CHECK-COUNT-4: arith.muli
func.func @canonical_one(%a: i32, %b: i32) -> (i32, i32) {
  %twiddle = arith.constant 32767 : i32
  %0, %1 = ondsp.cx_butterfly %a, %b, %twiddle {
    layout = #ondsp.cx_layout<packed_i16_imag_hi_real_lo>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    product = #ondsp.product<full>,
    product_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 15, rounding = nearest_even, overflow = saturate, saturate_to = i16>,
    output_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 1, rounding = nearest_even, overflow = saturate, saturate_to = i16>
  } : (i32, i32, i32) -> (i32, i32)
  return %0, %1 : i32, i32
}

func.func @canonical_minus_j(%a: i32, %b: i32) -> (i32, i32) {
  %twiddle = arith.constant -2147483648 : i32
  %0, %1 = ondsp.cx_butterfly %a, %b, %twiddle {
    layout = #ondsp.cx_layout<packed_i16_imag_hi_real_lo>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    product = #ondsp.product<full>,
    product_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 15, rounding = nearest_even, overflow = saturate, saturate_to = i16>,
    output_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 1, rounding = nearest_even, overflow = saturate, saturate_to = i16>
  } : (i32, i32, i32) -> (i32, i32)
  return %0, %1 : i32, i32
}

func.func @noncanonical_constant(%a: i32, %b: i32) -> (i32, i32) {
  %twiddle = arith.constant 1073758208 : i32
  %0, %1 = ondsp.cx_butterfly %a, %b, %twiddle {
    layout = #ondsp.cx_layout<packed_i16_imag_hi_real_lo>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    product = #ondsp.product<full>,
    product_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 15, rounding = nearest_even, overflow = saturate, saturate_to = i16>,
    output_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 1, rounding = nearest_even, overflow = saturate, saturate_to = i16>
  } : (i32, i32, i32) -> (i32, i32)
  return %0, %1 : i32, i32
}

func.func @runtime_twiddle(%a: i32, %b: i32, %twiddle: i32) -> (i32, i32) {
  %0, %1 = ondsp.cx_butterfly %a, %b, %twiddle {
    layout = #ondsp.cx_layout<packed_i16_imag_hi_real_lo>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    product = #ondsp.product<full>,
    product_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 15, rounding = nearest_even, overflow = saturate, saturate_to = i16>,
    output_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 1, rounding = nearest_even, overflow = saturate, saturate_to = i16>
  } : (i32, i32, i32) -> (i32, i32)
  return %0, %1 : i32, i32
}
