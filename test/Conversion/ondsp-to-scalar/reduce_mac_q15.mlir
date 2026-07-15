// RUN: ondrix-opt %s --convert-ondsp-q15-to-scalar | FileCheck %s

func.func @reduce_q15_saturate(
    %initial: !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>,
    %lhs: memref<8xi16>, %rhs: memref<8xi16>)
    -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate> {
  %result = ondsp.reduce_mac %initial, %lhs, %rhs {
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    product = #ondsp.product<full>
  } : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>, memref<8xi16>, memref<8xi16>) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  return %result : !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
}

// CHECK-LABEL: func.func @reduce_q15_saturate(%arg0: i40
// CHECK: %[[C0:.*]] = arith.constant 0 : index
// CHECK: %[[C8:.*]] = arith.constant 8 : index
// CHECK: %[[C1:.*]] = arith.constant 1 : index
// CHECK: %[[RESULT:.*]] = scf.for {{.*}} iter_args(%[[ACC:.*]] = %arg0) -> (i40) {
// CHECK: %[[LHS:.*]] = memref.load
// CHECK: %[[RHS:.*]] = memref.load
// CHECK: %[[LHS_EXT:.*]] = arith.extsi %[[LHS]] : i16 to i32
// CHECK: %[[RHS_EXT:.*]] = arith.extsi %[[RHS]] : i16 to i32
// CHECK: %[[PRODUCT:.*]] = arith.muli %[[LHS_EXT]], %[[RHS_EXT]] : i32
// CHECK: arith.extsi %[[ACC]] : i40 to i41
// CHECK: arith.extsi %[[PRODUCT]] : i32 to i41
// CHECK: arith.cmpi slt
// CHECK: arith.cmpi sgt
// CHECK: scf.yield {{.*}} : i40
// CHECK: return %[[RESULT]] : i40
// CHECK-NOT: ondsp.

func.func @reduce_q15_dynamic_wrap(
    %initial: !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>,
    %lhs: memref<?xi16>, %rhs: memref<?xi16>)
    -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap> {
  %result = ondsp.reduce_mac %initial, %lhs, %rhs {
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    product = #ondsp.product<full>
  } : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>, memref<?xi16>, memref<?xi16>) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>
  return %result : !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>
}

// CHECK-LABEL: func.func @reduce_q15_dynamic_wrap
// CHECK: %[[C0:.*]] = arith.constant 0 : index
// CHECK: %[[LHS_LEN:.*]] = memref.dim %arg1, %[[C0]] : memref<?xi16>
// CHECK: %[[RHS_LEN:.*]] = memref.dim %arg2, %[[C0]] : memref<?xi16>
// CHECK: %[[MATCH:.*]] = arith.cmpi eq, %[[LHS_LEN]], %[[RHS_LEN]] : index
// CHECK: cf.assert %[[MATCH]], "ondsp.reduce_mac requires equal operand lengths"
// CHECK: scf.for
// CHECK: arith.trunci {{.*}} : i41 to i40
// CHECK-NOT: ondsp.
