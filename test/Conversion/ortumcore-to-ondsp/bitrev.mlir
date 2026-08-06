// RUN: ondrix-opt %s --convert-ortumcore-to-ondsp-emulation | FileCheck %s

// The emulation realizes rev32 as the five-step masked butterfly swap in
// plain arith; one step's masks discriminate the constant set.
// CHECK-LABEL: func.func @bitrev_walk_step(
// CHECK-DAG: arith.constant 1431655765 : i32
// CHECK-DAG: arith.constant 858993459 : i32
// CHECK-DAG: arith.constant 252645135 : i32
// CHECK-DAG: arith.constant 16711935 : i32
// CHECK: arith.addi
// CHECK: arith.shrui
// CHECK: arith.ori
// CHECK-NOT: ortumcore.
func.func @bitrev_walk_step(%address: i32, %step: i32) -> i32 {
  %next = ortumcore.bitrev_add %address, %step : (i32, i32) -> i32
  return %next : i32
}

// CHECK-LABEL: func.func @bitrev_walk_back(
// CHECK: arith.subi
// CHECK-NOT: ortumcore.
func.func @bitrev_walk_back(%address: i32, %step: i32) -> i32 {
  %previous = ortumcore.bitrev_sub %address, %step : (i32, i32) -> i32
  return %previous : i32
}
