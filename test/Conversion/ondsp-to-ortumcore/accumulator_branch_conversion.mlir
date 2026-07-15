// RUN: ondrix-opt %s --convert-ondsp-to-ortumcore | FileCheck %s
// RUN: ondrix-opt %s --convert-ondsp-to-ortumcore | FileCheck %s --check-prefix=NO-SOURCE

func.func @branch_accumulator(
    %input: !ondsp.acc<storage = i40, frac = 30, signed>, %a: i16, %b: i16)
    -> !ondsp.acc<storage = i40, frac = 30, signed> {
  cf.br ^accumulate(%input : !ondsp.acc<storage = i40, frac = 30, signed>)
^accumulate(%acc: !ondsp.acc<storage = i40, frac = 30, signed>):
  %0 = ondsp.mac %acc, %a, %b {numeric = #ondsp.fixed<signed, storage = i16, frac = 15>, product = #ondsp.product<full>} : (!ondsp.acc<storage = i40, frac = 30, signed>, i16, i16) -> !ondsp.acc<storage = i40, frac = 30, signed>
  return %0 : !ondsp.acc<storage = i40, frac = 30, signed>
}

// CHECK-LABEL: func.func @branch_accumulator(
// CHECK-SAME: %[[INPUT:.*]]: !ortumcore.acc
// CHECK: cf.br ^[[ACCUMULATE:.*]](%[[INPUT]] : !ortumcore.acc)
// CHECK: ^[[ACCUMULATE]](%[[ACC:.*]]: !ortumcore.acc):
// CHECK: %[[MAC:.*]] = ortumcore.mac_add %[[ACC]], {{.*}} : (!ortumcore.acc, i16, i16) -> !ortumcore.acc
// CHECK: return %[[MAC]] : !ortumcore.acc
// NO-SOURCE-NOT: !ondsp.acc
