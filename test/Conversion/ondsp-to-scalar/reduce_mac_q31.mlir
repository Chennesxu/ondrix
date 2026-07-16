// RUN: ondrix-opt %s --convert-ondsp-fixed-to-scalar | FileCheck %s

func.func @q31_full_reduce(
    %initial: !ondsp.acc<storage = i64, frac = 62, signed, update_overflow = saturate>,
    %lhs: memref<4xi32>, %rhs: memref<4xi32>)
    -> !ondsp.acc<storage = i64, frac = 62, signed, update_overflow = saturate> {
  %result = ondsp.reduce_mac %initial, %lhs, %rhs {
    numeric = #ondsp.fixed<signed, storage = i32, frac = 31>,
    product = #ondsp.product<full>
  } : (!ondsp.acc<storage = i64, frac = 62, signed, update_overflow = saturate>, memref<4xi32>, memref<4xi32>) -> !ondsp.acc<storage = i64, frac = 62, signed, update_overflow = saturate>
  return %result : !ondsp.acc<storage = i64, frac = 62, signed, update_overflow = saturate>
}

func.func @q31_high_raw_reduce(
    %initial: !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>,
    %lhs: memref<?xi32>, %rhs: memref<?xi32>)
    -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap> {
  %result = ondsp.reduce_mac %initial, %lhs, %rhs {
    numeric = #ondsp.fixed<signed, storage = i32, frac = 31>,
    product = #ondsp.product<high_raw>
  } : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>, memref<?xi32>, memref<?xi32>) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>
  return %result : !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>
}

// CHECK-LABEL: func.func @q31_full_reduce(
// CHECK: %[[LOOP:.*]] = scf.for
// CHECK: %[[LHS:.*]] = memref.load
// CHECK: %[[RHS:.*]] = memref.load
// CHECK: %[[LHS_EXT:.*]] = arith.extsi %[[LHS]] : i32 to i64
// CHECK: %[[RHS_EXT:.*]] = arith.extsi %[[RHS]] : i32 to i64
// CHECK: %[[PRODUCT:.*]] = arith.muli %[[LHS_EXT]], %[[RHS_EXT]] : i64
// CHECK: arith.addi {{.*}} : i65
// CHECK: scf.yield
// CHECK: return %[[LOOP]] : i64

// CHECK-LABEL: func.func @q31_high_raw_reduce(
// CHECK: %[[LHS_LEN:.*]] = memref.dim
// CHECK: %[[RHS_LEN:.*]] = memref.dim
// CHECK: %[[MATCH:.*]] = arith.cmpi eq, %[[LHS_LEN]], %[[RHS_LEN]] : index
// CHECK: cf.assert %[[MATCH]], "ondsp.reduce_mac requires equal operand lengths"
// CHECK: %[[LOOP:.*]] = scf.for
// CHECK: %[[FULL:.*]] = arith.muli {{.*}} : i64
// CHECK: %[[SHIFTED:.*]] = arith.shrsi %[[FULL]], {{.*}} : i64
// CHECK: %[[HIGH:.*]] = arith.trunci %[[SHIFTED]] : i64 to i32
// CHECK: arith.addi {{.*}} : i41
// CHECK: return %[[LOOP]] : i40
// CHECK-NOT: ondsp.
