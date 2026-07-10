// RUN: ondrix-opt %s --convert-ondsp-to-ortumcore | FileCheck %s

func.func @q31_mac(%acc: !ondsp.acc<storage = i40, frac = 30, signed>, %a: i32, %b: i32)
    -> !ondsp.acc<storage = i40, frac = 30, signed> {
  %0 = ondsp.mac %acc, %a, %b {numeric = #ondsp.fixed<signed, storage = i32, frac = 31>, product = #ondsp.product<high>} : (!ondsp.acc<storage = i40, frac = 30, signed>, i32, i32) -> !ondsp.acc<storage = i40, frac = 30, signed>
  return %0 : !ondsp.acc<storage = i40, frac = 30, signed>
}

// CHECK-LABEL: func.func @q31_mac(
// CHECK-SAME: %[[ACC:.*]]: !ortumcore.acc, %[[A:.*]]: i32, %[[B:.*]]: i32) -> !ortumcore.acc {
// CHECK: %[[MAC:.*]] = ortumcore.qmac_add %[[ACC]], %[[A]], %[[B]] : (!ortumcore.acc, i32, i32) -> !ortumcore.acc
// CHECK: return %[[MAC]] : !ortumcore.acc
