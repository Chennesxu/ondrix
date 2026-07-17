// RUN: ondrix-opt %s --parallelize-ondsp-fixed-wrap-vector-reduce | FileCheck %s
// RUN: ondrix-opt %s --parallelize-ondsp-fixed-wrap-vector-reduce --normalize-ondsp-fixed-vector-reduce --convert-ondsp-fixed-to-scalar | FileCheck %s --check-prefix=FINAL

func.func @full_wrap(
    %initial: !ondsp.acc<storage = i64, frac = 62, signed, update_overflow = wrap>,
    %lhs: vector<4xi32>, %rhs: vector<4xi32>)
    -> !ondsp.acc<storage = i64, frac = 62, signed, update_overflow = wrap> {
  %result = ondsp.reduce_mac %initial, %lhs, %rhs {
    numeric = #ondsp.fixed<signed, storage = i32, frac = 31>,
    product = #ondsp.product<full>
  } : (!ondsp.acc<storage = i64, frac = 62, signed, update_overflow = wrap>, vector<4xi32>, vector<4xi32>) -> !ondsp.acc<storage = i64, frac = 62, signed, update_overflow = wrap>
  return %result : !ondsp.acc<storage = i64, frac = 62, signed, update_overflow = wrap>
}

// CHECK-LABEL: func.func @full_wrap
// CHECK: %[[LHS:.*]] = arith.extsi {{.*}} : vector<4xi32> to vector<4xi64>
// CHECK: %[[RHS:.*]] = arith.extsi {{.*}} : vector<4xi32> to vector<4xi64>
// CHECK: %[[TERMS:.*]] = arith.muli %[[LHS]], %[[RHS]] : vector<4xi64>
// CHECK: %[[SUM:.*]] = vector.reduction <add>, %[[TERMS]] : vector<4xi64> into i64
// CHECK: ondsp.acc_add_term {{.*}}, %[[SUM]] {term_numeric = #ondsp.fixed<signed, storage = i64, frac = 62>}
// CHECK-NOT: vector.extract
// CHECK-NOT: ondsp.reduce_mac

func.func @high_raw_wrap(
    %initial: !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>,
    %lhs: vector<4xi32>, %rhs: vector<4xi32>)
    -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap> {
  %result = ondsp.reduce_mac %initial, %lhs, %rhs {
    numeric = #ondsp.fixed<signed, storage = i32, frac = 31>,
    product = #ondsp.product<high_raw>
  } : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>, vector<4xi32>, vector<4xi32>) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>
  return %result : !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>
}

// CHECK-LABEL: func.func @high_raw_wrap
// CHECK: %[[FULL:.*]] = arith.muli {{.*}} : vector<4xi64>
// CHECK: %[[SHIFT:.*]] = arith.constant dense<32> : vector<4xi64>
// CHECK: %[[SHIFTED:.*]] = arith.shrsi %[[FULL]], %[[SHIFT]] : vector<4xi64>
// CHECK: %[[TERMS:.*]] = arith.trunci %[[SHIFTED]] : vector<4xi64> to vector<4xi32>
// CHECK: %[[WIDE:.*]] = arith.extsi %[[TERMS]] : vector<4xi32> to vector<4xi64>
// CHECK: %[[SUM:.*]] = vector.reduction <add>, %[[WIDE]] : vector<4xi64> into i64
// CHECK: ondsp.acc_add_term {{.*}}, %[[SUM]] {term_numeric = #ondsp.fixed<signed, storage = i64, frac = 30>}
// CHECK-NOT: vector.extract
// CHECK-NOT: ondsp.reduce_mac

func.func @full_saturate_preserves_order(
    %initial: !ondsp.acc<storage = i64, frac = 62, signed, update_overflow = saturate>,
    %lhs: vector<4xi32>, %rhs: vector<4xi32>)
    -> !ondsp.acc<storage = i64, frac = 62, signed, update_overflow = saturate> {
  %result = ondsp.reduce_mac %initial, %lhs, %rhs {
    numeric = #ondsp.fixed<signed, storage = i32, frac = 31>,
    product = #ondsp.product<full>
  } : (!ondsp.acc<storage = i64, frac = 62, signed, update_overflow = saturate>, vector<4xi32>, vector<4xi32>) -> !ondsp.acc<storage = i64, frac = 62, signed, update_overflow = saturate>
  return %result : !ondsp.acc<storage = i64, frac = 62, signed, update_overflow = saturate>
}

// CHECK-LABEL: func.func @full_saturate_preserves_order
// CHECK: ondsp.reduce_mac
// CHECK-NOT: vector.reduction

// FINAL-LABEL: func.func @full_wrap
// FINAL: vector.reduction <add>
// FINAL: arith.addi {{.*}} : i65
// FINAL-LABEL: func.func @high_raw_wrap
// FINAL: arith.shrsi {{.*}} : vector<4xi64>
// FINAL: vector.reduction <add>
// FINAL: arith.addi {{.*}} : i65
// FINAL-LABEL: func.func @full_saturate_preserves_order
// FINAL: vector.extract
// FINAL-NOT: ondsp.
