// RUN: ondrix-opt %s --convert-ondsp-fixed-to-scalar | FileCheck %s

func.func @mac_saturate(
    %acc: !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>,
    %lhs: i16, %rhs: i16)
    -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate> {
  %next = ondsp.mac %acc, %lhs, %rhs {
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    product = #ondsp.product<full>
  } : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>, i16, i16) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  return %next : !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
}

func.func @mac_sub_wrap(
    %acc: !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>,
    %lhs: i16, %rhs: i16)
    -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap> {
  %next = ondsp.mac_sub %acc, %lhs, %rhs {
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    product = #ondsp.product<full>
  } : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>, i16, i16) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>
  return %next : !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>
}

// CHECK-LABEL: func.func @mac_saturate(
// CHECK-SAME: %[[ACC:.*]]: i64, %[[LHS:.*]]: i16, %[[RHS:.*]]: i16) -> i64
// CHECK: %[[LHS_EXT:.*]] = arith.extsi %[[LHS]] : i16 to i32
// CHECK: %[[RHS_EXT:.*]] = arith.extsi %[[RHS]] : i16 to i32
// CHECK: %[[PRODUCT:.*]] = arith.muli %[[LHS_EXT]], %[[RHS_EXT]] : i32
// CHECK: %[[PRODUCT_EXT:.*]] = arith.extsi %[[PRODUCT]] : i32 to i64
// CHECK: %[[UPDATED:.*]] = arith.addi %[[ACC]], %[[PRODUCT_EXT]] : i64
// CHECK: %[[MIN:.*]] = arith.constant -549755813888 : i64
// CHECK: %[[MAX:.*]] = arith.constant 549755813887 : i64
// CHECK: %[[LOWER:.*]] = arith.maxsi %[[UPDATED]], %[[MIN]] : i64
// CHECK: %[[CLAMPED:.*]] = arith.minsi %[[LOWER]], %[[MAX]] : i64
// CHECK-NOT: arith.trunci
// CHECK: return %[[CLAMPED]] : i64

// A wrapping update multiplies in the i64 carrier: no i32 product, no second widening.
// CHECK-LABEL: func.func @mac_sub_wrap(
// CHECK-SAME: %[[ACC:.*]]: i64, %[[LHS:.*]]: i16, %[[RHS:.*]]: i16) -> i64
// CHECK: %[[LHS_EXT:.*]] = arith.extsi %[[LHS]] : i16 to i64
// CHECK: %[[RHS_EXT:.*]] = arith.extsi %[[RHS]] : i16 to i64
// CHECK: %[[PRODUCT:.*]] = arith.muli %[[LHS_EXT]], %[[RHS_EXT]] : i64
// CHECK-NOT: i32
// CHECK: %[[UPDATED:.*]] = arith.subi %[[ACC]], %[[PRODUCT]] : i64
// CHECK-NOT: arith.trunci
// CHECK: return %[[UPDATED]] : i64
// CHECK-NOT: ondsp.
