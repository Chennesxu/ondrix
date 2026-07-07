// RUN: ondrix-opt %s --convert-ondsp-to-ortumcore | FileCheck %s

func.func @q31_mac(%acc: !ortumcore.acc, %a: i32, %b: i32) -> !ortumcore.acc {
  %0 = ondsp.mac %acc, %a, %b {numeric = #ondsp.fixed<signed, storage = i32, frac = 31>} : (!ortumcore.acc, i32, i32) -> !ortumcore.acc
  return %0 : !ortumcore.acc
}

// CHECK-LABEL: func.func @q31_mac
// CHECK: ortumcore.qmac_add
// CHECK-SAME: (!ortumcore.acc, i32, i32) -> !ortumcore.acc
