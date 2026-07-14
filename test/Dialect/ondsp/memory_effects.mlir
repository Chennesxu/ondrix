// RUN: ondrix-opt %s --cse | FileCheck %s
// RUN: ondrix-opt %s --loop-invariant-code-motion | FileCheck %s --check-prefix=LICM

func.func @reduce_mac_reads_memory(%lhs: memref<8xf32>, %rhs: memref<8xf32>, %value: f32) -> (f32, f32) {
  %c0 = arith.constant 0 : index
  %before = ondsp.reduce_mac %lhs, %rhs {numeric = #ondsp.fp<format = f32, contract = off>} : (memref<8xf32>, memref<8xf32>) -> f32
  memref.store %value, %lhs[%c0] : memref<8xf32>
  %after = ondsp.reduce_mac %lhs, %rhs {numeric = #ondsp.fp<format = f32, contract = off>} : (memref<8xf32>, memref<8xf32>) -> f32
  return %before, %after : f32, f32
}

// CHECK-LABEL: func.func @reduce_mac_reads_memory
// CHECK: %[[BEFORE:.*]] = ondsp.reduce_mac
// CHECK: memref.store
// CHECK: %[[AFTER:.*]] = ondsp.reduce_mac
// CHECK: return %[[BEFORE]], %[[AFTER]]

func.func @reduce_mac_reads_unranked_first_operand(%lhs: memref<8xf32>, %rhs: f32, %value: f32) -> (f32, f32) {
  %c0 = arith.constant 0 : index
  %unranked = memref.cast %lhs : memref<8xf32> to memref<*xf32>
  %before = ondsp.reduce_mac %unranked, %rhs {numeric = #ondsp.fp<format = f32, contract = off>} : (memref<*xf32>, f32) -> f32
  memref.store %value, %lhs[%c0] : memref<8xf32>
  %after = ondsp.reduce_mac %unranked, %rhs {numeric = #ondsp.fp<format = f32, contract = off>} : (memref<*xf32>, f32) -> f32
  return %before, %after : f32, f32
}

// CHECK-LABEL: func.func @reduce_mac_reads_unranked_first_operand
// CHECK: %[[BEFORE:.*]] = ondsp.reduce_mac
// CHECK: memref.store
// CHECK: %[[AFTER:.*]] = ondsp.reduce_mac
// CHECK: return %[[BEFORE]], %[[AFTER]]

func.func @reduce_mac_reads_unranked_second_operand(%lhs: f32, %rhs: memref<8xf32>, %value: f32) -> (f32, f32) {
  %c0 = arith.constant 0 : index
  %unranked = memref.cast %rhs : memref<8xf32> to memref<*xf32>
  %before = ondsp.reduce_mac %lhs, %unranked {numeric = #ondsp.fp<format = f32, contract = off>} : (f32, memref<*xf32>) -> f32
  memref.store %value, %rhs[%c0] : memref<8xf32>
  %after = ondsp.reduce_mac %lhs, %unranked {numeric = #ondsp.fp<format = f32, contract = off>} : (f32, memref<*xf32>) -> f32
  return %before, %after : f32, f32
}

// CHECK-LABEL: func.func @reduce_mac_reads_unranked_second_operand
// CHECK: %[[BEFORE:.*]] = ondsp.reduce_mac
// CHECK: memref.store
// CHECK: %[[AFTER:.*]] = ondsp.reduce_mac
// CHECK: return %[[BEFORE]], %[[AFTER]]

func.func @scalar_reduce_mac_is_memory_effect_free(%lhs: i16, %rhs: i16) -> (i32, i32) {
  %first = ondsp.reduce_mac %lhs, %rhs {numeric = #ondsp.fixed<signed, storage = i16, frac = 15>, product = #ondsp.product<full>} : (i16, i16) -> i32
  %second = ondsp.reduce_mac %lhs, %rhs {numeric = #ondsp.fixed<signed, storage = i16, frac = 15>, product = #ondsp.product<full>} : (i16, i16) -> i32
  return %first, %second : i32, i32
}

// CHECK-LABEL: func.func @scalar_reduce_mac_is_memory_effect_free
// CHECK: %[[REDUCE:.*]] = ondsp.reduce_mac
// CHECK-NOT: ondsp.reduce_mac
// CHECK: return %[[REDUCE]], %[[REDUCE]]

func.func @scalar_reduce_mac_is_speculatable(%lhs: i16, %rhs: i16, %upper: index) -> i32 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %init = arith.constant 0 : i32
  %result = scf.for %i = %c0 to %upper step %c1 iter_args(%acc = %init) -> (i32) {
    %reduce = ondsp.reduce_mac %lhs, %rhs {numeric = #ondsp.fixed<signed, storage = i16, frac = 15>, product = #ondsp.product<full>} : (i16, i16) -> i32
    %next = arith.addi %acc, %reduce : i32
    scf.yield %next : i32
  }
  return %result : i32
}

// LICM-LABEL: func.func @scalar_reduce_mac_is_speculatable
// LICM: %[[REDUCE:.*]] = ondsp.reduce_mac
// LICM: scf.for
// LICM-NOT: ondsp.reduce_mac

func.func @unranked_reduce_mac_remains_in_loop(%lhs: memref<*xf32>, %rhs: memref<*xf32>, %upper: index) -> f32 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %init = arith.constant 0.0 : f32
  %result = scf.for %i = %c0 to %upper step %c1 iter_args(%acc = %init) -> (f32) {
    %reduce = ondsp.reduce_mac %lhs, %rhs {numeric = #ondsp.fp<format = f32, contract = off>} : (memref<*xf32>, memref<*xf32>) -> f32
    %next = arith.addf %acc, %reduce : f32
    scf.yield %next : f32
  }
  return %result : f32
}

// LICM-LABEL: func.func @unranked_reduce_mac_remains_in_loop
// LICM: scf.for
// LICM: ondsp.reduce_mac
