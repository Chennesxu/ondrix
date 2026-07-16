// RUN: ondrix-opt %s --convert-ortumcore-to-ondsp-emulation | FileCheck %s

func.func @loop(%lhs: i16, %rhs: i16, %count: index) -> !ortumcore.acc {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %zero = ortumcore.acc_init : !ortumcore.acc
  %result = scf.for %i = %c0 to %count step %c1
      iter_args(%current = %zero) -> (!ortumcore.acc) {
    %next = ortumcore.mac_add %current, %lhs, %rhs : (!ortumcore.acc, i16, i16) -> !ortumcore.acc
    scf.yield %next : !ortumcore.acc
  }
  return %result : !ortumcore.acc
}

// CHECK-LABEL: func.func @loop(
// CHECK-SAME: -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
// CHECK: %[[ZERO:.*]] = ondsp.acc_zero
// CHECK: %[[RESULT:.*]] = scf.for
// CHECK-SAME: iter_args(%{{.*}} = %[[ZERO]]) -> (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>)
// CHECK: ondsp.mac
// CHECK: scf.yield {{.*}} : !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
// CHECK: return %[[RESULT]]
// CHECK-NOT: ortumcore.
