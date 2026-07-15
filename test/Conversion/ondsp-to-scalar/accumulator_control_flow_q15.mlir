// RUN: ondrix-opt %s --convert-ondsp-q15-to-scalar | FileCheck %s

func.func @branch_accumulator(
    %input: !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>)
    -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate> {
  cf.br ^forward(%input : !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>)
^forward(%acc: !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>):
  return %acc : !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
}

func.func @conditional_accumulator(
    %condition: i1,
    %lhs: !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>,
    %rhs: !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>)
    -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap> {
  cf.cond_br %condition,
      ^left(%lhs : !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>),
      ^right(%rhs : !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>)
^left(%left_acc: !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>):
  cf.br ^merge(%left_acc : !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>)
^right(%right_acc: !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>):
  cf.br ^merge(%right_acc : !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>)
^merge(%merged_acc: !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>):
  return %merged_acc : !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>
}

func.func @select_accumulator(
    %condition: i1,
    %lhs: !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>,
    %rhs: !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>)
    -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate> {
  %selected = arith.select %condition, %lhs, %rhs
      : !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  return %selected : !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
}

// CHECK-LABEL: func.func @branch_accumulator(
// CHECK-SAME: %[[INPUT:.*]]: i40) -> i40
// CHECK: cf.br ^[[FORWARD:.*]](%[[INPUT]] : i40)
// CHECK: ^[[FORWARD]](%[[ACC:.*]]: i40):
// CHECK: return %[[ACC]] : i40

// CHECK-LABEL: func.func @conditional_accumulator(
// CHECK-SAME: %[[COND:.*]]: i1, %[[LHS:.*]]: i40, %[[RHS:.*]]: i40) -> i40
// CHECK: cf.cond_br %[[COND]], ^[[LEFT:.*]](%[[LHS]] : i40), ^[[RIGHT:.*]](%[[RHS]] : i40)
// CHECK: ^[[LEFT]](%[[LEFT_ACC:.*]]: i40):
// CHECK: cf.br ^[[MERGE:.*]](%[[LEFT_ACC]] : i40)
// CHECK: ^[[RIGHT]](%[[RIGHT_ACC:.*]]: i40):
// CHECK: cf.br ^[[MERGE]](%[[RIGHT_ACC]] : i40)
// CHECK: ^[[MERGE]](%[[MERGED:.*]]: i40):
// CHECK: return %[[MERGED]] : i40

// CHECK-LABEL: func.func @select_accumulator(
// CHECK-SAME: %[[COND:.*]]: i1, %[[LHS:.*]]: i40, %[[RHS:.*]]: i40) -> i40
// CHECK: %[[SELECTED:.*]] = arith.select %[[COND]], %[[LHS]], %[[RHS]] : i40
// CHECK: return %[[SELECTED]] : i40
// CHECK-NOT: !ondsp.acc
