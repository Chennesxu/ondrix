// RUN: ondrix-opt %s --convert-ondsp-to-ortumcore | FileCheck %s
// RUN: ondrix-opt %s --convert-ondsp-to-ortumcore | FileCheck %s --check-prefix=NO-SOURCE

func.func @accumulate(%acc: !ondsp.acc<storage = i40, frac = 30, signed>, %a: i16,
                      %b: i16) -> !ondsp.acc<storage = i40, frac = 30, signed> {
  %0 = ondsp.mac %acc, %a, %b {numeric = #ondsp.fixed<signed, storage = i16, frac = 15>, product = #ondsp.product<full>} : (!ondsp.acc<storage = i40, frac = 30, signed>, i16, i16) -> !ondsp.acc<storage = i40, frac = 30, signed>
  return %0 : !ondsp.acc<storage = i40, frac = 30, signed>
}

func.func @call_accumulate(%seed: i32, %a: i16, %b: i16) -> i32 {
  %0 = ondsp.acc_init %seed : (i32) -> !ondsp.acc<storage = i40, frac = 30, signed>
  %1 = func.call @accumulate(%0, %a, %b) : (!ondsp.acc<storage = i40, frac = 30, signed>, i16, i16) -> !ondsp.acc<storage = i40, frac = 30, signed>
  %2 = ondsp.acc_extract %1 : (!ondsp.acc<storage = i40, frac = 30, signed>) -> i32
  return %2 : i32
}

// CHECK-LABEL: func.func @accumulate(
// CHECK-SAME: %[[ACC:.*]]: !ortumcore.acc, %[[A:.*]]: i16, %[[B:.*]]: i16) -> !ortumcore.acc {
// CHECK: %[[MAC:.*]] = ortumcore.mac_add %[[ACC]], %[[A]], %[[B]] : (!ortumcore.acc, i16, i16) -> !ortumcore.acc
// CHECK: return %[[MAC]] : !ortumcore.acc
// CHECK-LABEL: func.func @call_accumulate(
// CHECK-SAME: %[[SEED:.*]]: i32, %[[CALL_A:.*]]: i16, %[[CALL_B:.*]]: i16) -> i32 {
// CHECK: %[[IMPORTED:.*]] = ortumcore.acc_import %[[SEED]] : (i32) -> !ortumcore.acc
// CHECK: %[[CALLED:.*]] = call @accumulate(%[[IMPORTED]], %[[CALL_A]], %[[CALL_B]]) : (!ortumcore.acc, i16, i16) -> !ortumcore.acc
// CHECK: %[[EXTRACTED:.*]] = ortumcore.acc_extract %[[CALLED]] : (!ortumcore.acc) -> i32
// CHECK: return %[[EXTRACTED]] : i32
// NO-SOURCE-NOT: !ondsp.acc
