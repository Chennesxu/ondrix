// RUN: ondrix-opt %s --convert-ondsp-fixed-to-scalar | FileCheck %s

func.func @make_zero()
    -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate> {
  %0 = ondsp.acc_zero : !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  return %0 : !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
}

// CHECK-LABEL: func.func @make_zero() -> i40
// CHECK: %[[ZERO:.*]] = arith.constant 0 : i40
// CHECK: return %[[ZERO]] : i40
// CHECK-NOT: ondsp.
