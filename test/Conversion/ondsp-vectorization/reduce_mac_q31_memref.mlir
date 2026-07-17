// RUN: ondrix-opt %s --vectorize-ondsp-fixed-memref-reduce="vector-width=4" | FileCheck %s

func.func @full_static_tail(
    %initial: !ondsp.acc<storage = i64, frac = 62, signed, update_overflow = saturate>,
    %lhs: memref<10xi32>, %rhs: memref<10xi32>)
    -> !ondsp.acc<storage = i64, frac = 62, signed, update_overflow = saturate> {
  %result = ondsp.reduce_mac %initial, %lhs, %rhs {
    numeric = #ondsp.fixed<signed, storage = i32, frac = 31>,
    product = #ondsp.product<full>
  } : (!ondsp.acc<storage = i64, frac = 62, signed, update_overflow = saturate>, memref<10xi32>, memref<10xi32>) -> !ondsp.acc<storage = i64, frac = 62, signed, update_overflow = saturate>
  return %result : !ondsp.acc<storage = i64, frac = 62, signed, update_overflow = saturate>
}

// CHECK-LABEL: func.func @full_static_tail
// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[C10:.*]] = arith.constant 10 : index
// CHECK-DAG: %[[C4:.*]] = arith.constant 4 : index
// CHECK: %[[REM:.*]] = arith.remui %[[C10]], %[[C4]] : index
// CHECK: %[[END:.*]] = arith.subi %[[C10]], %[[REM]] : index
// CHECK: %[[VACC:.*]] = scf.for %[[BASE:.*]] = %[[C0]] to %[[END]] step %[[C4]] iter_args
// CHECK: %[[LHS:.*]] = vector.load %{{.*}}[%[[BASE]]] : memref<10xi32>, vector<4xi32>
// CHECK: %[[RHS:.*]] = vector.load %{{.*}}[%[[BASE]]] : memref<10xi32>, vector<4xi32>
// CHECK: %[[NEXT:.*]] = ondsp.reduce_mac {{.*}}, %[[LHS]], %[[RHS]]
// CHECK: scf.yield %[[NEXT]]
// CHECK: scf.for %[[INDEX:.*]] = %[[END]] to %[[C10]] step %{{.*}} iter_args(%{{.*}} = %[[VACC]])
// CHECK: %[[LS:.*]] = memref.load %{{.*}}[%[[INDEX]]] : memref<10xi32>
// CHECK: %[[RS:.*]] = memref.load %{{.*}}[%[[INDEX]]] : memref<10xi32>
// CHECK: ondsp.mac {{.*}}, %[[LS]], %[[RS]]

func.func @high_raw_dynamic(
    %initial: !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>,
    %lhs: memref<?xi32>, %rhs: memref<?xi32>)
    -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap> {
  %result = ondsp.reduce_mac %initial, %lhs, %rhs {
    numeric = #ondsp.fixed<signed, storage = i32, frac = 31>,
    product = #ondsp.product<high_raw>
  } : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>, memref<?xi32>, memref<?xi32>) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>
  return %result : !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>
}

// CHECK-LABEL: func.func @high_raw_dynamic
// CHECK: %[[LHS_SIZE:.*]] = memref.dim
// CHECK: %[[RHS_SIZE:.*]] = memref.dim
// CHECK: %[[MATCH:.*]] = arith.cmpi eq, %[[LHS_SIZE]], %[[RHS_SIZE]] : index
// CHECK: cf.assert %[[MATCH]], "ondsp.reduce_mac requires equal operand lengths"
// CHECK: vector.load {{.*}} : memref<?xi32>, vector<4xi32>
// CHECK: ondsp.reduce_mac
// CHECK: ondsp.mac

func.func @unsupported_accumulator_fallback(
    %initial: !ondsp.acc<storage = i48, frac = 30, signed, update_overflow = saturate>,
    %lhs: memref<8xi32>, %rhs: memref<8xi32>)
    -> !ondsp.acc<storage = i48, frac = 30, signed, update_overflow = saturate> {
  %result = ondsp.reduce_mac %initial, %lhs, %rhs {
    numeric = #ondsp.fixed<signed, storage = i32, frac = 31>,
    product = #ondsp.product<high_raw>
  } : (!ondsp.acc<storage = i48, frac = 30, signed, update_overflow = saturate>, memref<8xi32>, memref<8xi32>) -> !ondsp.acc<storage = i48, frac = 30, signed, update_overflow = saturate>
  return %result : !ondsp.acc<storage = i48, frac = 30, signed, update_overflow = saturate>
}

// CHECK-LABEL: func.func @unsupported_accumulator_fallback
// CHECK: ondsp.reduce_mac
// CHECK-NOT: vector.load

func.func @dynamic_stride_fallback(
    %initial: !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>,
    %lhs: memref<?xi32, strided<[?], offset: ?>>,
    %rhs: memref<?xi32, strided<[?], offset: ?>>)
    -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate> {
  %result = ondsp.reduce_mac %initial, %lhs, %rhs {
    numeric = #ondsp.fixed<signed, storage = i32, frac = 31>,
    product = #ondsp.product<high_raw>
  } : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>, memref<?xi32, strided<[?], offset: ?>>, memref<?xi32, strided<[?], offset: ?>>) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  return %result : !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
}

// CHECK-LABEL: func.func @dynamic_stride_fallback
// CHECK: ondsp.reduce_mac
// CHECK-NOT: vector.load
