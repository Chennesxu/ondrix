// RUN: ondrix-opt %s --convert-ondsp-to-ortumcore | FileCheck %s

func.func @q15_mac(%acc: !ortumcore.acc, %a: i16, %b: i16) -> !ortumcore.acc {
  %0 = ondsp.mac %acc, %a, %b {numeric = #ondsp.fixed<signed, storage = i16, frac = 15>} : (!ortumcore.acc, i16, i16) -> !ortumcore.acc
  return %0 : !ortumcore.acc
}

// CHECK-LABEL: func.func @q15_mac
// CHECK-NOT: ondsp.mac
// CHECK: ortumcore.dual_mac
// CHECK-SAME: (!ortumcore.acc, i16, i16) -> !ortumcore.acc
