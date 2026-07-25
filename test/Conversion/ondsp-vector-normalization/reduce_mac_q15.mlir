// RUN: ondrix-opt %s --normalize-ondsp-fixed-vector-reduce | FileCheck %s
// RUN: not ondrix-opt %s --normalize-ondsp-fixed-vector-reduce --convert-ondsp-fixed-to-scalar 2>&1 | FileCheck %s --check-prefix=FINAL-ERROR

// FINAL-ERROR: failed to legalize operation 'ondsp.reduce_mac'

func.func @reduce_q15_vector(
    %initial: !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>,
    %lhs: vector<4xi16>, %rhs: vector<4xi16>)
    -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate> {
  %result = ondsp.reduce_mac %initial, %lhs, %rhs {
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    product = #ondsp.product<full>
  } : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>, vector<4xi16>, vector<4xi16>) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  return %result : !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
}

// CHECK-LABEL: func.func @reduce_q15_vector
// CHECK: %[[LHS:.*]] = arith.extsi {{.*}} : vector<4xi16> to vector<4xi32>
// CHECK: %[[RHS:.*]] = arith.extsi {{.*}} : vector<4xi16> to vector<4xi32>
// CHECK: %[[PRODUCTS:.*]] = arith.muli %[[LHS]], %[[RHS]] : vector<4xi32>
// CHECK: %[[P0:.*]] = vector.extract %[[PRODUCTS]][0] : vector<4xi32>
// CHECK: %[[A0:.*]] = ondsp.acc_add_term {{.*}}, %[[P0]]
// CHECK: %[[P1:.*]] = vector.extract %[[PRODUCTS]][1] : vector<4xi32>
// CHECK: %[[A1:.*]] = ondsp.acc_add_term %[[A0]], %[[P1]]
// CHECK: %[[P2:.*]] = vector.extract %[[PRODUCTS]][2] : vector<4xi32>
// CHECK: %[[A2:.*]] = ondsp.acc_add_term %[[A1]], %[[P2]]
// CHECK: %[[P3:.*]] = vector.extract %[[PRODUCTS]][3] : vector<4xi32>
// CHECK: %[[A3:.*]] = ondsp.acc_add_term %[[A2]], %[[P3]]
// CHECK: return %[[A3]]
// CHECK-NOT: ondsp.reduce_mac

func.func @preserve_unsupported_i65_vector(
    %initial: !ondsp.acc<storage = i65, frac = 30, signed, update_overflow = saturate>,
    %lhs: vector<4xi16>, %rhs: vector<4xi16>)
    -> !ondsp.acc<storage = i65, frac = 30, signed, update_overflow = saturate> {
  %result = ondsp.reduce_mac %initial, %lhs, %rhs {
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    product = #ondsp.product<full>
  } : (!ondsp.acc<storage = i65, frac = 30, signed, update_overflow = saturate>, vector<4xi16>, vector<4xi16>) -> !ondsp.acc<storage = i65, frac = 30, signed, update_overflow = saturate>
  return %result : !ondsp.acc<storage = i65, frac = 30, signed, update_overflow = saturate>
}

// CHECK-LABEL: func.func @preserve_unsupported_i65_vector
// CHECK: ondsp.reduce_mac
