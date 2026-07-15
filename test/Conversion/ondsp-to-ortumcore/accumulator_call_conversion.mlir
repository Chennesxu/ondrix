// RUN: ondrix-opt %s --convert-ondsp-to-ortumcore | FileCheck %s
// RUN: ondrix-opt %s --convert-ondsp-to-ortumcore | FileCheck %s --check-prefix=NO-SOURCE

func.func @accumulate(%acc: !ondsp.acc<storage = i40, frac = 30, signed>, %a: i16,
                      %b: i16) -> !ondsp.acc<storage = i40, frac = 30, signed> {
  %0 = ondsp.mac %acc, %a, %b {numeric = #ondsp.fixed<signed, storage = i16, frac = 15>, product = #ondsp.product<full>} : (!ondsp.acc<storage = i40, frac = 30, signed>, i16, i16) -> !ondsp.acc<storage = i40, frac = 30, signed>
  return %0 : !ondsp.acc<storage = i40, frac = 30, signed>
}

func.func @call_accumulate(%acc: !ondsp.acc<storage = i40, frac = 30, signed>,
                           %a: i16, %b: i16)
    -> !ondsp.acc<storage = i40, frac = 30, signed> {
  %0 = func.call @accumulate(%acc, %a, %b) : (!ondsp.acc<storage = i40, frac = 30, signed>, i16, i16) -> !ondsp.acc<storage = i40, frac = 30, signed>
  return %0 : !ondsp.acc<storage = i40, frac = 30, signed>
}

// CHECK-LABEL: func.func @accumulate(
// CHECK-SAME: %[[ACC:.*]]: !ortumcore.acc, %[[A:.*]]: i16, %[[B:.*]]: i16) -> !ortumcore.acc {
// CHECK: %[[MAC:.*]] = ortumcore.mac_add %[[ACC]], %[[A]], %[[B]] : (!ortumcore.acc, i16, i16) -> !ortumcore.acc
// CHECK: return %[[MAC]] : !ortumcore.acc
// CHECK-LABEL: func.func @call_accumulate(
// CHECK-SAME: %[[CALL_ACC:.*]]: !ortumcore.acc, %[[CALL_A:.*]]: i16, %[[CALL_B:.*]]: i16) -> !ortumcore.acc {
// CHECK: %[[CALLED:.*]] = call @accumulate(%[[CALL_ACC]], %[[CALL_A]], %[[CALL_B]]) : (!ortumcore.acc, i16, i16) -> !ortumcore.acc
// CHECK: return %[[CALLED]] : !ortumcore.acc
// NO-SOURCE-NOT: !ondsp.acc
