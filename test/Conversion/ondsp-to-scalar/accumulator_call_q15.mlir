// RUN: ondrix-opt %s --convert-ondsp-q15-to-scalar | FileCheck %s

func.func @forward(
    %acc: !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>)
    -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap> {
  return %acc : !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>
}

func.func @call_forward()
    -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap> {
  %zero = ondsp.acc_zero : !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>
  %result = func.call @forward(%zero) : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>
  return %result : !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>
}

// CHECK-LABEL: func.func @forward(
// CHECK-SAME: %[[ACC:.*]]: i40) -> i40
// CHECK: return %[[ACC]] : i40
// CHECK-LABEL: func.func @call_forward() -> i40
// CHECK: %[[ZERO:.*]] = arith.constant 0 : i40
// CHECK: %[[RESULT:.*]] = call @forward(%[[ZERO]]) : (i40) -> i40
// CHECK: return %[[RESULT]] : i40
// CHECK-NOT: ondsp.
