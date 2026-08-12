// RUN: ondrix-opt %s --convert-ondsp-cx-butterfly-to-ortumcore | FileCheck %s

// A target-inventory butterfly becomes the conjugated product plus the plain
// packed butterfly; the twiddle constant is conjugated exactly.
func.func @selects_floor_wrap(%a: i32, %b: i32) -> (i32, i32) {
  %tw = arith.constant 196610 : i32  // imag 3, real 2
  %0, %1 = ondsp.cx_butterfly %a, %b, %tw {
    layout = #ondsp.cx_layout<packed_i16_imag_hi_real_lo>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    product = #ondsp.product<full>,
    product_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 15, rounding = toward_negative, overflow = wrap, saturate_to = i16>,
    output_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 1, rounding = toward_negative, overflow = wrap, saturate_to = i16>
  } : (i32, i32, i32) -> (i32, i32)
  return %0, %1 : i32, i32
}

// CHECK-LABEL: func.func @selects_floor_wrap(
// CHECK-SAME: %[[A:.*]]: i32, %[[B:.*]]: i32)
// CHECK: %[[CONJ:.*]] = arith.constant -196606 : i32
// CHECK: %[[T:.*]] = ortumcore.cx_mul_conj %[[B]], %[[CONJ]]
// CHECK-SAME: layout = #ortumcore<cx_layout imag_hi>
// CHECK-SAME: overflow = #ortumcore<cx_overflow wrap>
// CHECK-SAME: rounding = #ortumcore<cx_rounding toward_negative>
// CHECK-SAME: shift = 15
// CHECK: %[[O0:.*]], %[[O1:.*]] = ortumcore.cx_bfly %[[A]], %[[T]]
// CHECK-SAME: shift = 1
// CHECK-SAME: variant = #ortumcore<cx_bfly_variant plain>
// CHECK: return %[[O0]], %[[O1]]
// CHECK-NOT: ondsp.cx_butterfly

func.func @selects_ntp_sat_shift0(%a: i32, %b: i32) -> (i32, i32) {
  %tw = arith.constant 1518511486 : i32  // 0x5A82A57E
  %0, %1 = ondsp.cx_butterfly %a, %b, %tw {
    layout = #ondsp.cx_layout<packed_i16_imag_hi_real_lo>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    product = #ondsp.product<full>,
    product_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 15, rounding = nearest_ties_positive, overflow = saturate, saturate_to = i16>,
    output_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 0, rounding = nearest_ties_positive, overflow = saturate, saturate_to = i16>
  } : (i32, i32, i32) -> (i32, i32)
  return %0, %1 : i32, i32
}

// CHECK-LABEL: func.func @selects_ntp_sat_shift0(
// CHECK: ortumcore.cx_mul_conj
// CHECK-SAME: rounding = #ortumcore<cx_rounding nearest_ties_positive>
// CHECK: ortumcore.cx_bfly
// CHECK-SAME: shift = 0
// CHECK-NOT: ondsp.cx_butterfly

// The default nearest_even profile is not a target capability: untouched.
func.func @keeps_nearest_even(%a: i32, %b: i32) -> (i32, i32) {
  %tw = arith.constant 196610 : i32
  %0, %1 = ondsp.cx_butterfly %a, %b, %tw {
    layout = #ondsp.cx_layout<packed_i16_imag_hi_real_lo>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    product = #ondsp.product<full>,
    product_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 15, rounding = nearest_even, overflow = saturate, saturate_to = i16>,
    output_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 1, rounding = nearest_even, overflow = saturate, saturate_to = i16>
  } : (i32, i32, i32) -> (i32, i32)
  return %0, %1 : i32, i32
}

// CHECK-LABEL: func.func @keeps_nearest_even(
// CHECK: ondsp.cx_butterfly
// CHECK-NOT: ortumcore.

// A runtime twiddle has no exact conjugation constant: untouched.
func.func @keeps_runtime_twiddle(%a: i32, %b: i32, %tw: i32) -> (i32, i32) {
  %0, %1 = ondsp.cx_butterfly %a, %b, %tw {
    layout = #ondsp.cx_layout<packed_i16_imag_hi_real_lo>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    product = #ondsp.product<full>,
    product_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 15, rounding = toward_negative, overflow = wrap, saturate_to = i16>,
    output_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 1, rounding = toward_negative, overflow = wrap, saturate_to = i16>
  } : (i32, i32, i32) -> (i32, i32)
  return %0, %1 : i32, i32
}

// CHECK-LABEL: func.func @keeps_runtime_twiddle(
// CHECK: ondsp.cx_butterfly
// CHECK-NOT: ortumcore.

// imag = -32768 has no representable exact conjugate: untouched.
func.func @keeps_unconjugatable_twiddle(%a: i32, %b: i32) -> (i32, i32) {
  %tw = arith.constant -2147483647 : i32  // 0x80000001
  %0, %1 = ondsp.cx_butterfly %a, %b, %tw {
    layout = #ondsp.cx_layout<packed_i16_imag_hi_real_lo>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    product = #ondsp.product<full>,
    product_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 15, rounding = toward_negative, overflow = wrap, saturate_to = i16>,
    output_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 1, rounding = toward_negative, overflow = wrap, saturate_to = i16>
  } : (i32, i32, i32) -> (i32, i32)
  return %0, %1 : i32, i32
}

// CHECK-LABEL: func.func @keeps_unconjugatable_twiddle(
// CHECK: ondsp.cx_butterfly
// CHECK-NOT: ortumcore.
