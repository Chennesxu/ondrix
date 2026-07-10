// RUN: ondrix-opt %s --convert-ondsp-to-ortumcore | FileCheck %s

func.func @branch_accumulator(%seed: i32, %a: i16, %b: i16) -> i32 {
  %0 = ondsp.acc_init %seed : (i32) -> !ondsp.acc<storage = i40, frac = 30, signed>
  cf.br ^accumulate(%0 : !ondsp.acc<storage = i40, frac = 30, signed>)
^accumulate(%acc: !ondsp.acc<storage = i40, frac = 30, signed>):
  %1 = ondsp.mac %acc, %a, %b {numeric = #ondsp.fixed<signed, storage = i16, frac = 15>, product = #ondsp.product<full>} : (!ondsp.acc<storage = i40, frac = 30, signed>, i16, i16) -> !ondsp.acc<storage = i40, frac = 30, signed>
  %2 = ondsp.acc_extract %1 : (!ondsp.acc<storage = i40, frac = 30, signed>) -> i32
  return %2 : i32
}

// CHECK-LABEL: func.func @branch_accumulator(
// CHECK: %[[IMPORTED:.*]] = ortumcore.acc_import %{{.*}} : (i32) -> !ortumcore.acc
// CHECK: cf.br ^[[ACCUMULATE:.*]](%[[IMPORTED]] : !ortumcore.acc)
// CHECK: ^[[ACCUMULATE]](%[[ACC:.*]]: !ortumcore.acc):
// CHECK: %[[MAC:.*]] = ortumcore.mac_add %[[ACC]], {{.*}} : (!ortumcore.acc, i16, i16) -> !ortumcore.acc
// CHECK: %[[EXTRACTED:.*]] = ortumcore.acc_extract %[[MAC]] : (!ortumcore.acc) -> i32
// CHECK: return %[[EXTRACTED]] : i32
