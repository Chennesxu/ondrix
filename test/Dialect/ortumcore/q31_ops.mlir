// RUN: ondrix-opt %s | ondrix-opt | FileCheck %s

// The Q31 raw-high MAC family shares the parameterless accumulator with the
// Q15 family: the raw high half lands at that accumulator's own frac position.
// CHECK-LABEL: func.func @q31_mac_web
// CHECK: ortumcore.acc_init : !ortumcore.acc
// CHECK: ortumcore.q31_mac_add {{.*}} : (!ortumcore.acc, i32, i32) -> !ortumcore.acc
// CHECK: ortumcore.q31_mac_sub {{.*}} : (!ortumcore.acc, i32, i32) -> !ortumcore.acc
// CHECK: ortumcore.acc_out {{.*}} {shift = 0 : i64}
func.func @q31_mac_web(%lhs: i32, %rhs: i32) -> i32 {
  %acc = ortumcore.acc_init : !ortumcore.acc
  %added = ortumcore.q31_mac_add %acc, %lhs, %rhs : (!ortumcore.acc, i32, i32) -> !ortumcore.acc
  %subtracted = ortumcore.q31_mac_sub %added, %rhs, %lhs
      : (!ortumcore.acc, i32, i32) -> !ortumcore.acc
  %out = ortumcore.acc_out %subtracted {shift = 0 : i64} : (!ortumcore.acc) -> i32
  return %out : i32
}

// Both scaling directions of the saturating add/sub, including the shift-0
// form that is a pure saturating add.
// CHECK-LABEL: func.func @scaled_saturating_pair
// CHECK: ortumcore.sat_shift_add {{.*}} {shift = 0 : i64} : (i32, i32) -> i32
// CHECK: ortumcore.sat_shift_sub {{.*}} {shift = 1 : i64} : (i32, i32) -> i32
// CHECK: ortumcore.sat_shift_add {{.*}} {shift = 3 : i64} : (i32, i32) -> i32
func.func @scaled_saturating_pair(%lhs: i32, %rhs: i32) -> i32 {
  %doubled = ortumcore.sat_shift_add %lhs, %lhs {shift = 0 : i64} : (i32, i32) -> i32
  %staged = ortumcore.sat_shift_sub %rhs, %doubled {shift = 1 : i64} : (i32, i32) -> i32
  %scaled = ortumcore.sat_shift_add %staged, %rhs {shift = 3 : i64} : (i32, i32) -> i32
  return %scaled : i32
}
