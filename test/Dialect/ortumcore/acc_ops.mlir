// RUN: ondrix-opt %s | FileCheck %s

// CHECK-LABEL: func.func @acc_ops
func.func @acc_ops(%a: i16, %b: i16) -> !ortumcore.acc {
  // CHECK: !ortumcore.acc
  %acc0 = ortumcore.acc_init : !ortumcore.acc

  // CHECK: ortumcore.mac_add
  %acc1 = ortumcore.mac_add %acc0, %a, %b : (!ortumcore.acc, i16, i16) -> !ortumcore.acc
  // CHECK: ortumcore.mac_sub
  %acc2 = ortumcore.mac_sub %acc1, %a, %b : (!ortumcore.acc, i16, i16) -> !ortumcore.acc
  return %acc2 : !ortumcore.acc
}

// CHECK-LABEL: func.func @pair_types
func.func @pair_types(%acc_pair: !ortumcore.acc_pair, %fp: !ortumcore.fp, %fp_pair: !ortumcore.fp_pair) {
  // CHECK: !ortumcore.acc_pair
  // CHECK: !ortumcore.fp_pair
  return
}
