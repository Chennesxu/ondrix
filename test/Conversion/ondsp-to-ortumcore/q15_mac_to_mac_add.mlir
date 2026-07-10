// RUN: ondrix-opt %s --convert-ondsp-to-ortumcore | FileCheck %s

func.func @q15_mac(%acc: !ondsp.acc<storage = i40, frac = 30, signed>, %a: i16, %b: i16)
    -> !ondsp.acc<storage = i40, frac = 30, signed> {
  %0 = ondsp.mac %acc, %a, %b {numeric = #ondsp.fixed<signed, storage = i16, frac = 15>, product = #ondsp.product<full>} : (!ondsp.acc<storage = i40, frac = 30, signed>, i16, i16) -> !ondsp.acc<storage = i40, frac = 30, signed>
  return %0 : !ondsp.acc<storage = i40, frac = 30, signed>
}

// CHECK-LABEL: func.func @q15_mac(
// CHECK-SAME: %[[ACC:.*]]: !ortumcore.acc, %[[A:.*]]: i16, %[[B:.*]]: i16) -> !ortumcore.acc {
// CHECK-NOT: ondsp.mac
// CHECK: %[[MAC:.*]] = ortumcore.mac_add %[[ACC]], %[[A]], %[[B]] : (!ortumcore.acc, i16, i16) -> !ortumcore.acc
// CHECK: return %[[MAC]] : !ortumcore.acc
