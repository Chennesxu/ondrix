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
// CHECK-SAME: %[[ACC:.*]]: i40, %[[LHS:.*]]: i16, %[[RHS:.*]]: i16) -> i40
// CHECK: %[[LHS_EXT:.*]] = arith.extsi %[[LHS]] : i16 to i32
// CHECK: %[[RHS_EXT:.*]] = arith.extsi %[[RHS]] : i16 to i32
// CHECK: %[[PRODUCT:.*]] = arith.muli %[[LHS_EXT]], %[[RHS_EXT]] : i32
// CHECK: %[[ACC_EXT:.*]] = arith.extsi %[[ACC]] : i40 to i41
// CHECK: %[[PRODUCT_EXT:.*]] = arith.extsi %[[PRODUCT]] : i32 to i41
// CHECK: %[[UPDATED:.*]] = arith.addi %[[ACC_EXT]], %[[PRODUCT_EXT]] : i41
// CHECK: %[[MIN:.*]] = arith.constant -549755813888 : i41
// CHECK: %[[MAX:.*]] = arith.constant 549755813887 : i41
// CHECK: %[[LOWER:.*]] = arith.maxsi %[[UPDATED]], %[[MIN]] : i41
// CHECK: %[[CLAMPED:.*]] = arith.minsi %[[LOWER]], %[[MAX]] : i41
// CHECK: %[[RESULT:.*]] = arith.trunci %[[CLAMPED]] : i41 to i40
// CHECK: return %[[RESULT]] : i40

// CHECK-LABEL: func.func @mac_sub_wrap(
// CHECK-SAME: %[[ACC:.*]]: i40, %[[LHS:.*]]: i16, %[[RHS:.*]]: i16) -> i40
// CHECK: %[[LHS_EXT:.*]] = arith.extsi %[[LHS]] : i16 to i32
// CHECK: %[[RHS_EXT:.*]] = arith.extsi %[[RHS]] : i16 to i32
// CHECK: %[[PRODUCT:.*]] = arith.muli %[[LHS_EXT]], %[[RHS_EXT]] : i32
// CHECK: %[[ACC_EXT:.*]] = arith.extsi %[[ACC]] : i40 to i41
// CHECK: %[[PRODUCT_EXT:.*]] = arith.extsi %[[PRODUCT]] : i32 to i41
// CHECK: %[[UPDATED:.*]] = arith.subi %[[ACC_EXT]], %[[PRODUCT_EXT]] : i41
// CHECK: %[[RESULT:.*]] = arith.trunci %[[UPDATED]] : i41 to i40
// CHECK: return %[[RESULT]] : i40
// CHECK-NOT: ondsp.
