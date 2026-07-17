// RUN: ondrix-opt %s --normalize-ondsp-fixed-vector-reduce | FileCheck %s

func.func @full_ordered(
    %initial: !ondsp.acc<storage = i64, frac = 62, signed, update_overflow = saturate>,
    %lhs: vector<4xi32>, %rhs: vector<4xi32>)
    -> !ondsp.acc<storage = i64, frac = 62, signed, update_overflow = saturate> {
  %result = ondsp.reduce_mac %initial, %lhs, %rhs {
    numeric = #ondsp.fixed<signed, storage = i32, frac = 31>,
    product = #ondsp.product<full>
  } : (!ondsp.acc<storage = i64, frac = 62, signed, update_overflow = saturate>, vector<4xi32>, vector<4xi32>) -> !ondsp.acc<storage = i64, frac = 62, signed, update_overflow = saturate>
  return %result : !ondsp.acc<storage = i64, frac = 62, signed, update_overflow = saturate>
}

// CHECK-LABEL: func.func @full_ordered
// CHECK: %[[LHS:.*]] = arith.extsi {{.*}} : vector<4xi32> to vector<4xi64>
// CHECK: %[[RHS:.*]] = arith.extsi {{.*}} : vector<4xi32> to vector<4xi64>
// CHECK: %[[TERMS:.*]] = arith.muli %[[LHS]], %[[RHS]] : vector<4xi64>
// CHECK: %[[T0:.*]] = vector.extract %[[TERMS]][0] : vector<4xi64>
// CHECK: %[[A0:.*]] = ondsp.acc_add_term {{.*}}, %[[T0]] {term_numeric = #ondsp.fixed<signed, storage = i64, frac = 62>}
// CHECK: %[[T1:.*]] = vector.extract %[[TERMS]][1] : vector<4xi64>
// CHECK: %[[A1:.*]] = ondsp.acc_add_term %[[A0]], %[[T1]]
// CHECK: %[[T2:.*]] = vector.extract %[[TERMS]][2] : vector<4xi64>
// CHECK: %[[A2:.*]] = ondsp.acc_add_term %[[A1]], %[[T2]]
// CHECK: %[[T3:.*]] = vector.extract %[[TERMS]][3] : vector<4xi64>
// CHECK: ondsp.acc_add_term %[[A2]], %[[T3]]
// CHECK-NOT: ondsp.reduce_mac

func.func @high_raw_ordered(
    %initial: !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>,
    %lhs: vector<4xi32>, %rhs: vector<4xi32>)
    -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate> {
  %result = ondsp.reduce_mac %initial, %lhs, %rhs {
    numeric = #ondsp.fixed<signed, storage = i32, frac = 31>,
    product = #ondsp.product<high_raw>
  } : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>, vector<4xi32>, vector<4xi32>) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  return %result : !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
}

// CHECK-LABEL: func.func @high_raw_ordered
// CHECK: %[[LHS:.*]] = arith.extsi {{.*}} : vector<4xi32> to vector<4xi64>
// CHECK: %[[RHS:.*]] = arith.extsi {{.*}} : vector<4xi32> to vector<4xi64>
// CHECK: %[[FULL:.*]] = arith.muli %[[LHS]], %[[RHS]] : vector<4xi64>
// CHECK: %[[SHIFT:.*]] = arith.constant dense<32> : vector<4xi64>
// CHECK: %[[SHIFTED:.*]] = arith.shrsi %[[FULL]], %[[SHIFT]] : vector<4xi64>
// CHECK: %[[TERMS:.*]] = arith.trunci %[[SHIFTED]] : vector<4xi64> to vector<4xi32>
// CHECK: %[[T0:.*]] = vector.extract %[[TERMS]][0] : vector<4xi32>
// CHECK: %[[A0:.*]] = ondsp.acc_add_term {{.*}}, %[[T0]] {term_numeric = #ondsp.fixed<signed, storage = i32, frac = 30>}
// CHECK: %[[T1:.*]] = vector.extract %[[TERMS]][1] : vector<4xi32>
// CHECK: %[[A1:.*]] = ondsp.acc_add_term %[[A0]], %[[T1]]
// CHECK: %[[T2:.*]] = vector.extract %[[TERMS]][2] : vector<4xi32>
// CHECK: %[[A2:.*]] = ondsp.acc_add_term %[[A1]], %[[T2]]
// CHECK: %[[T3:.*]] = vector.extract %[[TERMS]][3] : vector<4xi32>
// CHECK: ondsp.acc_add_term %[[A2]], %[[T3]]
// CHECK-NOT: ondsp.reduce_mac

func.func @unsupported_accumulator_preserved(
    %initial: !ondsp.acc<storage = i48, frac = 30, signed, update_overflow = saturate>,
    %lhs: vector<4xi32>, %rhs: vector<4xi32>)
    -> !ondsp.acc<storage = i48, frac = 30, signed, update_overflow = saturate> {
  %result = ondsp.reduce_mac %initial, %lhs, %rhs {
    numeric = #ondsp.fixed<signed, storage = i32, frac = 31>,
    product = #ondsp.product<high_raw>
  } : (!ondsp.acc<storage = i48, frac = 30, signed, update_overflow = saturate>, vector<4xi32>, vector<4xi32>) -> !ondsp.acc<storage = i48, frac = 30, signed, update_overflow = saturate>
  return %result : !ondsp.acc<storage = i48, frac = 30, signed, update_overflow = saturate>
}

// CHECK-LABEL: func.func @unsupported_accumulator_preserved
// CHECK: ondsp.reduce_mac
