// RUN: ondrix-opt %s --convert-ondsp-to-ortumcore | FileCheck %s
// RUN: ondrix-opt %s --convert-ondsp-to-ortumcore | FileCheck %s --check-prefix=NO-SOURCE

func.func @forward_accumulator(%acc: !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>)
    -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate> {
  return %acc : !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
}

func.func @call_forward_accumulator(%acc: !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>)
    -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate> {
  %0 = func.call @forward_accumulator(%acc) : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  return %0 : !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
}

// CHECK-LABEL: func.func @forward_accumulator(
// CHECK-SAME: %[[ACC:.*]]: !ortumcore.acc) -> !ortumcore.acc {
// CHECK: return %[[ACC]] : !ortumcore.acc
// CHECK-LABEL: func.func @call_forward_accumulator(
// CHECK-SAME: %[[CALL_ACC:.*]]: !ortumcore.acc) -> !ortumcore.acc {
// CHECK: %[[CALLED:.*]] = call @forward_accumulator(%[[CALL_ACC]]) : (!ortumcore.acc) -> !ortumcore.acc
// CHECK: return %[[CALLED]] : !ortumcore.acc
// NO-SOURCE-NOT: !ondsp.acc
