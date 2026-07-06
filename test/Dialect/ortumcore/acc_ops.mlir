// RUN: ondrix-opt %s | FileCheck %s

// CHECK-LABEL: func.func @mr_ops
func.func @mr_ops(%a: i16, %b: i16) -> i32 {
  // CHECK: !ortumcore.acc
  %mr0 = ortumcore.acc_init : !ortumcore.acc

  // CHECK: ortumcore.dual_mac
  %mr1 = ortumcore.dual_mac %mr0, %a, %b : (!ortumcore.acc, i16, i16) -> !ortumcore.acc
  %x = ortumcore.acc_extract %mr1 : (!ortumcore.acc) -> i32
  return %x : i32
}
