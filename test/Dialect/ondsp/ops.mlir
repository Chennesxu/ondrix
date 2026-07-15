// RUN: ondrix-opt %s | FileCheck %s

// CHECK-LABEL: func.func @ondsp_ops
func.func @ondsp_ops(%a: i16, %b: i16, %v0: vector<4xi16>, %v1: vector<4xi16>,
                     %p0: i32, %p1: i32, %tw: i32, %i: index, %s: index)
    -> (i16, !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>, i32, index) {
  // CHECK: ondsp.acc_zero
  %zero = ondsp.acc_zero : !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  // CHECK: ondsp.acc_import
  %imported = ondsp.acc_import %a {src = #ondsp.fixed<signed, storage = i16, frac = 15>} : (i16) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>

  // CHECK: ondsp.mac
  %mac = ondsp.mac %imported, %a, %b {numeric = #ondsp.fixed<signed, storage = i16, frac = 15>, product = #ondsp.product<full>} : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>, i16, i16) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>

  // CHECK: ondsp.acc_export
  %exported = ondsp.acc_export %mac {dst = #ondsp.fixed<signed, storage = i16, frac = 15>, rounding = #ondsp.rounding<nearest_even>, overflow = #ondsp.overflow<saturate>} : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>) -> i16

  // CHECK: ondsp.reduce_mac
  %red = ondsp.reduce_mac %zero, %v0, %v1 {numeric = #ondsp.fixed<signed, storage = i16, frac = 15>, product = #ondsp.product<full>} : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>, vector<4xi16>, vector<4xi16>) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>

  // CHECK: ondsp.cx_butterfly
  // CHECK: scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 15, rounding = toward_negative, overflow = saturate, saturate_to = i16>
  %o0, %o1 = ondsp.cx_butterfly %p0, %p1, %tw {layout = #ondsp.cx_layout<packed_i16_imag_hi_real_lo>, numeric = #ondsp.fixed<signed, storage = i16, frac = 15>, product = #ondsp.product<full>, scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 15, rounding = toward_negative, overflow = saturate, saturate_to = i16>} : (i32, i32, i32) -> (i32, i32)

  // CHECK: ondsp.bitrev_add
  %j = ondsp.bitrev_add %i, %s {width = 10 : i64} : (index, index) -> index
  return %exported, %red, %o0, %j : i16, !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>, i32, index
}
