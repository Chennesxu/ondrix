// RUN: ondrix-opt %s --vectorize-ondsp-fixed-memref-reduce="vector-width=4" | FileCheck %s

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

func.func @nonzero_integer_memory_space_fallback(
    %initial: !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>,
    %lhs: memref<8xi16, 1>, %rhs: memref<8xi16, 1>)
    -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate> {
  %result = ondsp.reduce_mac %initial, %lhs, %rhs {
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    product = #ondsp.product<full>
  } : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>, memref<8xi16, 1>, memref<8xi16, 1>) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  return %result : !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
}

// CHECK-LABEL: func.func @nonzero_integer_memory_space_fallback
// CHECK: ondsp.reduce_mac
// CHECK-NOT: vector.load

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

func.func @wider_than_i64_accumulator_fallback(
    %initial: !ondsp.acc<storage = i65, frac = 30, signed, update_overflow = wrap>,
    %lhs: memref<8xi16>, %rhs: memref<8xi16>)
    -> !ondsp.acc<storage = i65, frac = 30, signed, update_overflow = wrap> {
  %result = ondsp.reduce_mac %initial, %lhs, %rhs {
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    product = #ondsp.product<full>
  } : (!ondsp.acc<storage = i65, frac = 30, signed, update_overflow = wrap>, memref<8xi16>, memref<8xi16>) -> !ondsp.acc<storage = i65, frac = 30, signed, update_overflow = wrap>
  return %result : !ondsp.acc<storage = i65, frac = 30, signed, update_overflow = wrap>
}

// CHECK-LABEL: func.func @wider_than_i64_accumulator_fallback
// CHECK: ondsp.reduce_mac
// CHECK-NOT: vector.load

// -----

// A genuine convolution reverses its kernel in the LAYOUT, because reduce_mac
// pairs both operands in increasing index order. The chunk reads that view by
// loading its span forward and reversing the lanes.
// CHECK-LABEL: func.func @reversed_kernel_view
// CHECK: %[[SUB:.*]] = memref.subview %{{.*}}[31] [32] [-1]
// CHECK: scf.for
// The span ends at origin 31 minus the last lane, so the forward base is 28.
// CHECK: %[[END:.*]] = arith.constant 28 : index
// CHECK: %[[SPAN:.*]] = arith.subi %[[END]], %{{.*}} : index
// CHECK: vector.load %{{.*}}[%[[SPAN]]] : memref<32xi16>, vector<4xi16>
// CHECK: vector.shuffle %{{.*}} [3, 2, 1, 0] : vector<4xi16>, vector<4xi16>
func.func @reversed_kernel_view(%window: memref<32xi16>, %kernel: memref<32xi16>)
    -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap> {
  %seed = ondsp.acc_zero : !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>
  %view = memref.subview %kernel[31] [32] [-1]
      : memref<32xi16> to memref<32xi16, strided<[-1], offset: 31>>
  %cast = memref.cast %view
      : memref<32xi16, strided<[-1], offset: 31>> to memref<?xi16, strided<[-1], offset: ?>>
  %win = memref.cast %window : memref<32xi16> to memref<?xi16>
  %r = ondsp.reduce_mac %seed, %win, %cast {
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>, product = #ondsp.product<full>
  } : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>,
       memref<?xi16>, memref<?xi16, strided<[-1], offset: ?>>)
      -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>
  return %r : !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>
}

// -----

// A stride of -2 is not a reversal this pass can undo: the span it would load
// is not the set the ordered schedule read.
// CHECK-LABEL: func.func @strided_view_refused
// CHECK-NOT: vector.shuffle
func.func @strided_view_refused(%window: memref<32xi16>, %kernel: memref<64xi16>)
    -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap> {
  %seed = ondsp.acc_zero : !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>
  %view = memref.subview %kernel[63] [32] [-2]
      : memref<64xi16> to memref<32xi16, strided<[-2], offset: 63>>
  %cast = memref.cast %view
      : memref<32xi16, strided<[-2], offset: 63>> to memref<?xi16, strided<[-2], offset: ?>>
  %win = memref.cast %window : memref<32xi16> to memref<?xi16>
  %r = ondsp.reduce_mac %seed, %win, %cast {
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>, product = #ondsp.product<full>
  } : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>,
       memref<?xi16>, memref<?xi16, strided<[-2], offset: ?>>)
      -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>
  return %r : !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>
}
