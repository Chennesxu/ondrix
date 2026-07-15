// RUN: ondrix-opt %s --convert-ondsp-to-ortumcore | FileCheck %s

func.func @acc_zero() -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate> {
  %0 = ondsp.acc_zero : !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  return %0 : !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
}

// CHECK-LABEL: func.func @acc_zero() -> !ortumcore.acc
// CHECK: %[[ZERO:.*]] = ortumcore.acc_init : !ortumcore.acc
// CHECK: return %[[ZERO]] : !ortumcore.acc
