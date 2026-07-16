// RUN: ondrix-opt %s --convert-ortumcore-to-ondsp-emulation | FileCheck %s

func.func @accumulator(%lhs: i16, %rhs: i16) -> !ortumcore.acc {
  %zero = ortumcore.acc_init : !ortumcore.acc
  %added = ortumcore.mac_add %zero, %lhs, %rhs : (!ortumcore.acc, i16, i16) -> !ortumcore.acc
  %result = ortumcore.mac_sub %added, %lhs, %rhs : (!ortumcore.acc, i16, i16) -> !ortumcore.acc
  return %result : !ortumcore.acc
}

// CHECK-LABEL: func.func @accumulator(
// CHECK-SAME: %[[LHS:.*]]: i16, %[[RHS:.*]]: i16) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
// CHECK: %[[ZERO:.*]] = ondsp.acc_zero
// CHECK: %[[ADDED:.*]] = ondsp.mac %[[ZERO]], %[[LHS]], %[[RHS]]
// CHECK-SAME: numeric = #ondsp.fixed<signed, storage = i16, frac = 15>
// CHECK-SAME: product = #ondsp.product<full>
// CHECK: %[[RESULT:.*]] = ondsp.mac_sub %[[ADDED]], %[[LHS]], %[[RHS]]
// CHECK: return %[[RESULT]]
// CHECK-NOT: ortumcore.
