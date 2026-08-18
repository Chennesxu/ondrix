// RUN: ondrix-opt %s --convert-ortumcore-to-ondsp-emulation | FileCheck %s --implicit-check-not=ortumcore.

// The Q31 MAC family emulates as the same ondsp accumulator updates as the Q15
// family with one attribute changed — the raw-high product selection — which is
// the whole difference between the two capabilities.
// CHECK-LABEL: func.func @q31_web
// CHECK: ondsp.acc_zero
// CHECK: ondsp.mac {{.*}}numeric = #ondsp.fixed<signed, storage = i32, frac = 31>, product = #ondsp.product<high_raw>
// CHECK: ondsp.mac_sub {{.*}}numeric = #ondsp.fixed<signed, storage = i32, frac = 31>, product = #ondsp.product<high_raw>
// The readout keeps the accumulator's frac-30 position: it is a shift of zero.
// CHECK: ondsp.acc_export {{.*}}dst = #ondsp.fixed<signed, storage = i32, frac = 30>
// CHECK-SAME: overflow = #ondsp.overflow<saturate>
// CHECK-SAME: rounding = #ondsp.rounding<toward_negative>
func.func @q31_web(%lhs: i32, %rhs: i32) -> i32 {
  %acc = ortumcore.acc_init : !ortumcore.acc
  %added = ortumcore.q31_mac_add %acc, %lhs, %rhs : (!ortumcore.acc, i32, i32) -> !ortumcore.acc
  %subtracted = ortumcore.q31_mac_sub %added, %rhs, %lhs
      : (!ortumcore.acc, i32, i32) -> !ortumcore.acc
  %out = ortumcore.acc_out %subtracted {shift = 0 : i64} : (!ortumcore.acc) -> i32
  return %out : i32
}

// The scaled saturating add/sub is one declared ondsp shift boundary over the
// exact sum: the operation's shift becomes the boundary's right shift, and its
// floor and saturation are the only rounding and overflow it can realize.
// CHECK-LABEL: func.func @scaled_saturating
// CHECK: ondsp.add_shift
// CHECK-SAME: #ondsp.scale<pre_shift_left = 0, post_shift_right = 0, rounding = toward_negative, overflow = saturate, saturate_to = i32>
// CHECK: ondsp.sub_shift
// CHECK-SAME: #ondsp.scale<pre_shift_left = 0, post_shift_right = 1, rounding = toward_negative, overflow = saturate, saturate_to = i32>
func.func @scaled_saturating(%lhs: i32, %rhs: i32) -> i32 {
  %doubled = ortumcore.sat_shift_add %lhs, %lhs {shift = 0 : i64} : (i32, i32) -> i32
  %staged = ortumcore.sat_shift_sub %rhs, %doubled {shift = 1 : i64} : (i32, i32) -> i32
  return %staged : i32
}
