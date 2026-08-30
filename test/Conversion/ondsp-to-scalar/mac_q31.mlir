// RUN: ondrix-opt %s --convert-ondsp-fixed-to-scalar | FileCheck %s

func.func @q31_full_saturate(
    %acc: !ondsp.acc<storage = i64, frac = 62, signed, update_overflow = saturate>,
    %lhs: i32, %rhs: i32)
    -> !ondsp.acc<storage = i64, frac = 62, signed, update_overflow = saturate> {
  %next = ondsp.mac %acc, %lhs, %rhs {
    numeric = #ondsp.fixed<signed, storage = i32, frac = 31>,
    product = #ondsp.product<full>
  } : (!ondsp.acc<storage = i64, frac = 62, signed, update_overflow = saturate>, i32, i32) -> !ondsp.acc<storage = i64, frac = 62, signed, update_overflow = saturate>
  return %next : !ondsp.acc<storage = i64, frac = 62, signed, update_overflow = saturate>
}

func.func @q31_high_raw_wrap(
    %acc: !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>,
    %lhs: i32, %rhs: i32)
    -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap> {
  %next = ondsp.mac_sub %acc, %lhs, %rhs {
    numeric = #ondsp.fixed<signed, storage = i32, frac = 31>,
    product = #ondsp.product<high_raw>
  } : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>, i32, i32) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>
  return %next : !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>
}

func.func @q15_in_mixed_module(
    %acc: !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>,
    %lhs: i16, %rhs: i16)
    -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate> {
  %next = ondsp.mac %acc, %lhs, %rhs {
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    product = #ondsp.product<full>
  } : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>, i16, i16) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  return %next : !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
}

// CHECK-LABEL: func.func @q31_full_saturate(
// CHECK-SAME: %[[ACC:.*]]: i64, %[[LHS:.*]]: i32, %[[RHS:.*]]: i32) -> i64
// CHECK: %[[LHS_EXT:.*]] = arith.extsi %[[LHS]] : i32 to i64
// CHECK: %[[RHS_EXT:.*]] = arith.extsi %[[RHS]] : i32 to i64
// CHECK: %[[PRODUCT:.*]] = arith.muli %[[LHS_EXT]], %[[RHS_EXT]] : i64
// CHECK: %[[ACC_EXT:.*]] = arith.extsi %[[ACC]] : i64 to i65
// CHECK: %[[PRODUCT_EXT:.*]] = arith.extsi %[[PRODUCT]] : i64 to i65
// CHECK: %[[UPDATED:.*]] = arith.addi %[[ACC_EXT]], %[[PRODUCT_EXT]] : i65
// CHECK: %[[MIN:.*]] = arith.constant -9223372036854775808 : i65
// CHECK: %[[MAX:.*]] = arith.constant 9223372036854775807 : i65
// CHECK: %[[LOWER:.*]] = arith.maxsi %[[UPDATED]], %[[MIN]] : i65
// CHECK: arith.minsi %[[LOWER]], %[[MAX]] : i65
// CHECK: %[[RESULT:.*]] = arith.trunci {{.*}} : i65 to i64
// CHECK: return %[[RESULT]] : i64

// CHECK-LABEL: func.func @q31_high_raw_wrap(
// CHECK-SAME: %[[ACC:.*]]: i40, %[[LHS:.*]]: i32, %[[RHS:.*]]: i32) -> i40
// CHECK: %[[LHS_EXT:.*]] = arith.extsi %[[LHS]] : i32 to i64
// CHECK: %[[RHS_EXT:.*]] = arith.extsi %[[RHS]] : i32 to i64
// CHECK: %[[FULL:.*]] = arith.muli %[[LHS_EXT]], %[[RHS_EXT]] : i64
// CHECK: %[[SHIFT:.*]] = arith.constant 32 : i64
// CHECK: %[[SHIFTED:.*]] = arith.shrsi %[[FULL]], %[[SHIFT]] : i64
// CHECK: %[[HIGH:.*]] = arith.trunci %[[SHIFTED]] : i64 to i32
// CHECK: %[[ACC_EXT:.*]] = arith.extsi %[[ACC]] : i40 to i41
// CHECK: %[[HIGH_EXT:.*]] = arith.extsi %[[HIGH]] : i32 to i41
// CHECK: %[[UPDATED:.*]] = arith.subi %[[ACC_EXT]], %[[HIGH_EXT]] : i41
// CHECK: %[[RESULT:.*]] = arith.trunci %[[UPDATED]] : i41 to i40
// CHECK: return %[[RESULT]] : i40

// CHECK-LABEL: func.func @q15_in_mixed_module(
// CHECK-SAME: %{{.*}}: i40, %{{.*}}: i16, %{{.*}}: i16) -> i40
// CHECK-NOT: ondsp.
