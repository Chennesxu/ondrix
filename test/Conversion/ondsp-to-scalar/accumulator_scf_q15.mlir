// RUN: ondrix-opt %s --convert-ondsp-fixed-to-scalar | FileCheck %s

func.func @for_accumulator(%lower: index, %upper: index, %step: index,
    %initial: !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>,
    %lhs: i16, %rhs: i16) -> i16 {
  %result = scf.for %i = %lower to %upper step %step
      iter_args(%acc = %initial)
      -> (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>) {
    %next = ondsp.mac %acc, %lhs, %rhs {
      numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
      product = #ondsp.product<full>
    } : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>, i16, i16) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
    scf.yield %next : !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  }
  %exported = ondsp.acc_export %result {
    dst = #ondsp.fixed<signed, storage = i16, frac = 15>,
    rounding = #ondsp.rounding<toward_negative>,
    overflow = #ondsp.overflow<saturate>
  } : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>) -> i16
  return %exported : i16
}

func.func @if_accumulator(%condition: i1,
    %lhs: !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>,
    %rhs: !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>)
    -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap> {
  %result = scf.if %condition
      -> (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>) {
    scf.yield %lhs : !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>
  } else {
    scf.yield %rhs : !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>
  }
  return %result : !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>
}

func.func @while_accumulator(%condition: i1,
    %initial: !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>)
    -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate> {
  %result = scf.while (%acc = %initial)
      : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>)
      -> (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>) {
    scf.condition(%condition) %acc
        : !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  } do {
  ^bb0(%acc: !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>):
    scf.yield %acc : !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  }
  return %result : !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
}

// CHECK-LABEL: func.func @for_accumulator(
// CHECK-SAME: %[[LOWER:.*]]: index, %[[UPPER:.*]]: index, %[[STEP:.*]]: index,
// CHECK-SAME: %[[INITIAL:.*]]: i40, %[[LHS:.*]]: i16, %[[RHS:.*]]: i16) -> i16
// CHECK: %[[RESULT:.*]] = scf.for %{{.*}} = %[[LOWER]] to %[[UPPER]] step %[[STEP]]
// CHECK-SAME: iter_args(%[[ACC:.*]] = %[[INITIAL]]) -> (i40) {
// CHECK: scf.yield %{{.*}} : i40
// CHECK: return %{{.*}} : i16

// CHECK-LABEL: func.func @if_accumulator(
// CHECK: %[[RESULT:.*]] = scf.if %{{.*}} -> (i40) {
// CHECK: scf.yield %{{.*}} : i40
// CHECK: } else {
// CHECK: scf.yield %{{.*}} : i40
// CHECK: return %[[RESULT]] : i40

// CHECK-LABEL: func.func @while_accumulator(
// CHECK: %[[RESULT:.*]] = scf.while (%[[ACC:.*]] = %{{.*}}) : (i40) -> i40 {
// CHECK: scf.condition(%{{.*}}) %[[ACC]] : i40
// CHECK: } do {
// CHECK: ^bb0(%[[BODY_ACC:.*]]: i40):
// CHECK: scf.yield %[[BODY_ACC]] : i40
// CHECK: return %[[RESULT]] : i40
// CHECK-NOT: !ondsp.acc
