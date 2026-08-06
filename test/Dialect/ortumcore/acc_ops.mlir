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

// CHECK-LABEL: func.func @acc_out
func.func @acc_out(%acc: !ortumcore.acc) -> (i32, i32) {
  // CHECK: ortumcore.acc_out %{{.*}} {shift = 0 : i64}
  %raw = ortumcore.acc_out %acc {shift = 0 : i64} : (!ortumcore.acc) -> i32
  // CHECK: ortumcore.acc_out %{{.*}} {shift = 15 : i64}
  %scaled = ortumcore.acc_out %acc {shift = 15 : i64} : (!ortumcore.acc) -> i32
  return %raw, %scaled : i32, i32
}

// CHECK-LABEL: func.func @bitrev_ops(
// CHECK: ortumcore.bitrev_add %{{.*}}, %{{.*}} : (i32, i32) -> i32
// CHECK: ortumcore.bitrev_sub %{{.*}}, %{{.*}} : (i32, i32) -> i32
func.func @bitrev_ops(%address: i32, %step: i32) -> i32 {
  %up = ortumcore.bitrev_add %address, %step : (i32, i32) -> i32
  %down = ortumcore.bitrev_sub %up, %step : (i32, i32) -> i32
  return %down : i32
}
