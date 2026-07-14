// RUN: ondrix-opt %s --convert-ondsp-to-ortumcore | FileCheck %s

func.func @acc_zero() -> !ondsp.acc<storage = i40, frac = 30, signed> {
  %0 = ondsp.acc_zero : !ondsp.acc<storage = i40, frac = 30, signed>
  return %0 : !ondsp.acc<storage = i40, frac = 30, signed>
}

// CHECK-LABEL: func.func @acc_zero() -> !ortumcore.acc
// CHECK: %[[ZERO:.*]] = ortumcore.acc_init : !ortumcore.acc
// CHECK: return %[[ZERO]] : !ortumcore.acc
