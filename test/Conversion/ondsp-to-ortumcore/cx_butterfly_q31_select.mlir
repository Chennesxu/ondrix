// RUN: ondrix-opt %s --convert-ondsp-cx-butterfly-to-ortumcore | FileCheck %s
// RUN: ondrix-opt %s --convert-ondsp-cx-butterfly-to-ortumcore | FileCheck %s --check-prefix=CLOSED

// Pins the decomposed shape of the raw-high selection — accumulator webs,
// saturating doubling, scaled stage legs; the argument is in Passes.td.

// CHECK-LABEL: func.func @raw_high_q31
// The real term: one web that adds br*wr and subtracts bi*wi.
// CHECK: ortumcore.acc_init
// CHECK: ortumcore.q31_mac_add
// CHECK: ortumcore.q31_mac_sub
// CHECK: ortumcore.acc_out %{{.*}} {shift = 0 : i64}
// CHECK: ortumcore.sat_shift_add %[[REAL:.*]], %[[REAL]] {shift = 0 : i64}
// The imaginary term: one web that adds both of its cross products.
// CHECK: ortumcore.acc_init
// CHECK: ortumcore.q31_mac_add
// CHECK: ortumcore.q31_mac_add
// CHECK: ortumcore.acc_out %{{.*}} {shift = 0 : i64}
// CHECK: ortumcore.sat_shift_add %[[IMAG:.*]], %[[IMAG]] {shift = 0 : i64}
// The stage combine carries the declared output shift on both legs.
// CHECK: ortumcore.sat_shift_add %{{.*}} {shift = 1 : i64}
// CHECK: ortumcore.sat_shift_add %{{.*}} {shift = 1 : i64}
// CHECK: ortumcore.sat_shift_sub %{{.*}} {shift = 1 : i64}
// CHECK: ortumcore.sat_shift_sub %{{.*}} {shift = 1 : i64}
// CHECK-NOT: ondsp.cx_butterfly
// CHECK-LABEL: func.func @full_product_q31
func.func @raw_high_q31(%a: i64, %b: i64, %tw: i64) -> (i64, i64) {
  %o0, %o1 = ondsp.cx_butterfly %a, %b, %tw {
    layout = #ondsp.cx_layout<packed_i32_imag_hi_real_lo>,
    numeric = #ondsp.fixed<signed, storage = i32, frac = 31>,
    product = #ondsp.product<high_raw>,
    product_scale = #ondsp.scale<pre_shift_left = 1, post_shift_right = 0, rounding = toward_negative, overflow = saturate, saturate_to = i32>,
    output_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 1, rounding = toward_negative, overflow = saturate, saturate_to = i32>
  } : (i64, i64, i64) -> (i64, i64)
  return %o0, %o1 : i64, i64
}

// The full-product Q31 profile is a different equation, and the target computes
// only the raw-high one: it must stay generic rather than pick up a sequence
// that would round somewhere else.
// CLOSED-LABEL: func.func @full_product_q31
// CLOSED: ondsp.cx_butterfly
// CLOSED-NOT: ortumcore.
func.func @full_product_q31(%a: i64, %b: i64, %tw: i64) -> (i64, i64) {
  %o0, %o1 = ondsp.cx_butterfly %a, %b, %tw {
    layout = #ondsp.cx_layout<packed_i32_imag_hi_real_lo>,
    numeric = #ondsp.fixed<signed, storage = i32, frac = 31>,
    product = #ondsp.product<full>,
    product_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 31, rounding = nearest_even, overflow = saturate, saturate_to = i32>,
    output_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 1, rounding = nearest_even, overflow = saturate, saturate_to = i32>
  } : (i64, i64, i64) -> (i64, i64)
  return %o0, %o1 : i64, i64
}
