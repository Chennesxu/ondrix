// RUN: ondrix-opt %s --vectorize-ondsp-q15-memref-reduce="vector-width=4" | FileCheck %s

func.func @static_tail(
    %initial: !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>,
    %lhs: memref<10xi16>, %rhs: memref<10xi16>)
    -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate> {
  %result = ondsp.reduce_mac %initial, %lhs, %rhs {
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    product = #ondsp.product<full>
  } : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>, memref<10xi16>, memref<10xi16>) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  return %result : !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
}

// CHECK-LABEL: func.func @static_tail
// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[C10:.*]] = arith.constant 10 : index
// CHECK-DAG: %[[C4:.*]] = arith.constant 4 : index
// CHECK: %[[REM:.*]] = arith.remui %[[C10]], %[[C4]] : index
// CHECK: %[[END:.*]] = arith.subi %[[C10]], %[[REM]] : index
// CHECK: %[[VACC:.*]] = scf.for %[[BASE:.*]] = %[[C0]] to %[[END]] step %[[C4]] iter_args(%[[CURRENT:.*]] = %{{.*}})
// CHECK: %[[LHS:.*]] = vector.load %{{.*}}[%[[BASE]]] : memref<10xi16>, vector<4xi16>
// CHECK: %[[RHS:.*]] = vector.load %{{.*}}[%[[BASE]]] : memref<10xi16>, vector<4xi16>
// CHECK: %[[NEXT:.*]] = ondsp.reduce_mac %[[CURRENT]], %[[LHS]], %[[RHS]]
// CHECK: scf.yield %[[NEXT]]
// CHECK: %[[C1:.*]] = arith.constant 1 : index
// CHECK: %[[RESULT:.*]] = scf.for %[[INDEX:.*]] = %[[END]] to %[[C10]] step %[[C1]] iter_args(%[[TAIL_ACC:.*]] = %[[VACC]])
// CHECK: %[[LS:.*]] = memref.load %{{.*}}[%[[INDEX]]] : memref<10xi16>
// CHECK: %[[RS:.*]] = memref.load %{{.*}}[%[[INDEX]]] : memref<10xi16>
// CHECK: %[[TAIL_NEXT:.*]] = ondsp.mac %[[TAIL_ACC]], %[[LS]], %[[RS]]
// CHECK: scf.yield %[[TAIL_NEXT]]
// CHECK: return %[[RESULT]]

func.func @dynamic(
    %initial: !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>,
    %lhs: memref<?xi16>, %rhs: memref<?xi16>)
    -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap> {
  %result = ondsp.reduce_mac %initial, %lhs, %rhs {
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    product = #ondsp.product<full>
  } : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>, memref<?xi16>, memref<?xi16>) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>
  return %result : !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>
}

// CHECK-LABEL: func.func @dynamic
// CHECK: %[[LHS_SIZE:.*]] = memref.dim
// CHECK: %[[RHS_SIZE:.*]] = memref.dim
// CHECK: %[[MATCH:.*]] = arith.cmpi eq, %[[LHS_SIZE]], %[[RHS_SIZE]] : index
// CHECK: cf.assert %[[MATCH]], "ondsp.reduce_mac requires equal operand lengths"
// CHECK: vector.load {{.*}} : memref<?xi16>, vector<4xi16>

func.func @strided_fallback(
    %initial: !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>,
    %lhs: memref<?xi16, strided<[?], offset: ?>>,
    %rhs: memref<?xi16, strided<[?], offset: ?>>)
    -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate> {
  %result = ondsp.reduce_mac %initial, %lhs, %rhs {
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    product = #ondsp.product<full>
  } : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>, memref<?xi16, strided<[?], offset: ?>>, memref<?xi16, strided<[?], offset: ?>>) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  return %result : !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
}

// CHECK-LABEL: func.func @strided_fallback
// CHECK: ondsp.reduce_mac
// CHECK-NOT: vector.load

func.func @unsupported_q31(
    %initial: !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>,
    %lhs: memref<8xi32>, %rhs: memref<8xi32>)
    -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate> {
  %result = ondsp.reduce_mac %initial, %lhs, %rhs {
    numeric = #ondsp.fixed<signed, storage = i32, frac = 31>,
    product = #ondsp.product<high_raw>
  } : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>, memref<8xi32>, memref<8xi32>) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  return %result : !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
}

// CHECK-LABEL: func.func @unsupported_q31
// CHECK: ondsp.reduce_mac
// CHECK-NOT: vector.load

func.func @integer_memory_space_vectorized(
    %initial: !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>,
    %lhs: memref<8xi16, 1>, %rhs: memref<8xi16, 1>)
    -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate> {
  %result = ondsp.reduce_mac %initial, %lhs, %rhs {
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    product = #ondsp.product<full>
  } : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>, memref<8xi16, 1>, memref<8xi16, 1>) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  return %result : !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
}

// CHECK-LABEL: func.func @integer_memory_space_vectorized
// CHECK: vector.load {{.*}} : memref<8xi16, 1>, vector<4xi16>

func.func @lhs_custom_memory_space_fallback(
    %initial: !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>,
    %lhs: memref<8xi16, "device">, %rhs: memref<8xi16>)
    -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate> {
  %result = ondsp.reduce_mac %initial, %lhs, %rhs {
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    product = #ondsp.product<full>
  } : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>, memref<8xi16, "device">, memref<8xi16>) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  return %result : !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
}

// CHECK-LABEL: func.func @lhs_custom_memory_space_fallback
// CHECK: ondsp.reduce_mac
// CHECK-NOT: vector.load

func.func @rhs_custom_memory_space_fallback(
    %initial: !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>,
    %lhs: memref<8xi16>, %rhs: memref<8xi16, "device">)
    -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate> {
  %result = ondsp.reduce_mac %initial, %lhs, %rhs {
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    product = #ondsp.product<full>
  } : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>, memref<8xi16>, memref<8xi16, "device">) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  return %result : !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
}

// CHECK-LABEL: func.func @rhs_custom_memory_space_fallback
// CHECK: ondsp.reduce_mac
// CHECK-NOT: vector.load
