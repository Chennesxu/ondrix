// RUN: ondrix-opt %s --convert-ondsp-fixed-to-scalar --split-input-file | FileCheck %s

// The packed-Q15 product carrier is 33 bits because an ARBITRARY i16 twiddle
// makes the imaginary cross sum reach exactly 2^31. The carrier narrows to 32
// exactly when the twiddle is a constant the query proves cannot: both sums
// are bounded by 32768 * (|wr| + |wi|), so |wr| + |wi| <= 65535 suffices.

// The 45-degree twiddle, |wr| + |wi| = 46340 -- every quantized unit-modulus
// twiddle clears the bound by a factor of sqrt(2).
func.func @unit_modulus_narrows(%a: i32, %b: i32) -> (i32, i32) {
  %w = arith.constant 0xA57E5A82 : i32
  %0, %1 = ondsp.cx_butterfly %a, %b, %w {
    layout = #ondsp.cx_layout<packed_i16_imag_hi_real_lo>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    product = #ondsp.product<full>,
    product_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 15, rounding = nearest_even, overflow = saturate, saturate_to = i16>,
    output_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 1, rounding = nearest_even, overflow = saturate, saturate_to = i16>
  } : (i32, i32, i32) -> (i32, i32)
  return %0, %1 : i32, i32
}

// CHECK-LABEL: func.func @unit_modulus_narrows
// CHECK-COUNT-4: arith.extsi {{.*}} : i16 to i32
// CHECK-COUNT-4: arith.muli {{.*}} : i32
// CHECK-NOT: i33

// -----

// A CONSTANT twiddle that is not unit-modulus. The query is about the value,
// not about constant-ness, and this is the one pair that reaches 2^31.
func.func @overlong_constant_stays_wide(%a: i32, %b: i32) -> (i32, i32) {
  %w = arith.constant 0x80008000 : i32
  %0, %1 = ondsp.cx_butterfly %a, %b, %w {
    layout = #ondsp.cx_layout<packed_i16_imag_hi_real_lo>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    product = #ondsp.product<full>,
    product_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 15, rounding = nearest_even, overflow = saturate, saturate_to = i16>,
    output_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 1, rounding = nearest_even, overflow = saturate, saturate_to = i16>
  } : (i32, i32, i32) -> (i32, i32)
  return %0, %1 : i32, i32
}

// CHECK-LABEL: func.func @overlong_constant_stays_wide
// CHECK-COUNT-4: arith.extsi {{.*}} : i16 to i33
// CHECK-COUNT-4: arith.muli {{.*}} : i33

// -----

// A runtime twiddle carries no value at all, so it fails closed.
func.func @runtime_twiddle_stays_wide(%a: i32, %b: i32, %w: i32) -> (i32, i32) {
  %0, %1 = ondsp.cx_butterfly %a, %b, %w {
    layout = #ondsp.cx_layout<packed_i16_imag_hi_real_lo>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    product = #ondsp.product<full>,
    product_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 15, rounding = nearest_even, overflow = saturate, saturate_to = i16>,
    output_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 1, rounding = nearest_even, overflow = saturate, saturate_to = i16>
  } : (i32, i32, i32) -> (i32, i32)
  return %0, %1 : i32, i32
}

// CHECK-LABEL: func.func @runtime_twiddle_stays_wide
// CHECK-COUNT-4: arith.extsi {{.*}} : i16 to i33
