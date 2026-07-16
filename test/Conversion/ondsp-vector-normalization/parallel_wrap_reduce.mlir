// RUN: ondrix-opt %s --parallelize-ondsp-q15-wrap-vector-reduce | FileCheck %s
// RUN: ondrix-opt %s --parallelize-ondsp-q15-wrap-vector-reduce --normalize-ondsp-q15-vector-reduce --convert-ondsp-q15-to-scalar | FileCheck %s --check-prefix=FINAL

func.func @parallel_wrap(
    %initial: !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>,
    %lhs: vector<8xi16>, %rhs: vector<8xi16>)
    -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap> {
  %result = ondsp.reduce_mac %initial, %lhs, %rhs {
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    product = #ondsp.product<full>
  } : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>, vector<8xi16>, vector<8xi16>) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>
  return %result : !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>
}

// CHECK-LABEL: func.func @parallel_wrap
// CHECK: %[[LHS:.*]] = arith.extsi {{.*}} : vector<8xi16> to vector<8xi32>
// CHECK: %[[RHS:.*]] = arith.extsi {{.*}} : vector<8xi16> to vector<8xi32>
// CHECK: %[[PRODUCTS:.*]] = arith.muli %[[LHS]], %[[RHS]] : vector<8xi32>
// CHECK: %[[WIDE:.*]] = arith.extsi %[[PRODUCTS]] : vector<8xi32> to vector<8xi64>
// CHECK: %[[SUM:.*]] = vector.reduction <add>, %[[WIDE]] : vector<8xi64> into i64
// CHECK: ondsp.acc_add_term {{.*}}, %[[SUM]] {term_numeric = #ondsp.fixed<signed, storage = i64, frac = 30>}
// CHECK-NOT: vector.extract
// CHECK-NOT: ondsp.reduce_mac

func.func @preserve_saturate(
    %initial: !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>,
    %lhs: vector<8xi16>, %rhs: vector<8xi16>)
    -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate> {
  %result = ondsp.reduce_mac %initial, %lhs, %rhs {
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    product = #ondsp.product<full>
  } : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>, vector<8xi16>, vector<8xi16>) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  return %result : !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
}

// CHECK-LABEL: func.func @preserve_saturate
// CHECK: ondsp.reduce_mac
// CHECK-NOT: vector.reduction

// FINAL-LABEL: func.func @parallel_wrap
// FINAL: vector.reduction <add>
// FINAL: arith.extsi {{.*}} : i40 to i65
// FINAL: arith.extsi {{.*}} : i64 to i65
// FINAL: arith.addi
// FINAL-LABEL: func.func @preserve_saturate
// FINAL: vector.extract
// FINAL-NOT: ondsp.
