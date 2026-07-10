// RUN: ondrix-opt %s | FileCheck %s

// CHECK-LABEL: func.func @ondsp_ops
func.func @ondsp_ops(%a: i16, %b: i16, %p0: i32, %p1: i32, %tw: i32,
                     %i: index, %s: index) -> (i32, i32, i32, index) {
  %acc = ondsp.acc_init %a : (i16) -> !ondsp.acc<storage = i40, frac = 30, signed>

  // CHECK: ondsp.mac
  %mac = ondsp.mac %acc, %a, %b {numeric = #ondsp.fixed<signed, storage = i16, frac = 15>, product = #ondsp.product<full>} : (!ondsp.acc<storage = i40, frac = 30, signed>, i16, i16) -> !ondsp.acc<storage = i40, frac = 30, signed>
  %out = ondsp.acc_extract %mac {scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 15, rounding = trunc, overflow = saturate, saturate_to = i32>} : (!ondsp.acc<storage = i40, frac = 30, signed>) -> i32

  // CHECK: ondsp.reduce_mac
  %red = ondsp.reduce_mac %a, %b {numeric = #ondsp.fixed<signed, storage = i16, frac = 15>, product = #ondsp.product<full>} : (i16, i16) -> i32

  // CHECK: ondsp.cx_butterfly
  // CHECK: scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 15, rounding = trunc, overflow = saturate, saturate_to = i16>
  %o0, %o1 = ondsp.cx_butterfly %p0, %p1, %tw {layout = #ondsp.cx_layout<packed_i16_imag_hi_real_lo>, numeric = #ondsp.fixed<signed, storage = i16, frac = 15>, product = #ondsp.product<full>, scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 15, rounding = trunc, overflow = saturate, saturate_to = i16>} : (i32, i32, i32) -> (i32, i32)

  // CHECK: ondsp.bitrev_add
  %j = ondsp.bitrev_add %i, %s {width = 10 : i64} : (index, index) -> index
  return %out, %red, %o0, %j : i32, i32, i32, index
}
