// RUN: ondrix-opt %s --cse | FileCheck %s
// RUN: ondrix-opt %s --loop-invariant-code-motion | FileCheck %s --check-prefix=LICM

func.func @fir_reads_memory(%input: memref<8xf32>, %coeffs: memref<8xf32>, %value: f32) -> (f32, f32) {
  %c0 = arith.constant 0 : index
  %before = ondrix.fir %input, %coeffs {numeric = #ondsp.fp<format = f32, contract = off>} : (memref<8xf32>, memref<8xf32>) -> f32
  memref.store %value, %input[%c0] : memref<8xf32>
  %after = ondrix.fir %input, %coeffs {numeric = #ondsp.fp<format = f32, contract = off>} : (memref<8xf32>, memref<8xf32>) -> f32
  return %before, %after : f32, f32
}

// CHECK-LABEL: func.func @fir_reads_memory
// CHECK: %[[BEFORE:.*]] = ondrix.fir
// CHECK: memref.store
// CHECK: %[[AFTER:.*]] = ondrix.fir
// CHECK: return %[[BEFORE]], %[[AFTER]]

func.func @fir_reads_first_operand(%input: memref<8xf32>, %coeff: f32, %value: f32) -> (f32, f32) {
  %c0 = arith.constant 0 : index
  %before = ondrix.fir %input, %coeff {numeric = #ondsp.fp<format = f32, contract = off>} : (memref<8xf32>, f32) -> f32
  memref.store %value, %input[%c0] : memref<8xf32>
  %after = ondrix.fir %input, %coeff {numeric = #ondsp.fp<format = f32, contract = off>} : (memref<8xf32>, f32) -> f32
  return %before, %after : f32, f32
}

// CHECK-LABEL: func.func @fir_reads_first_operand
// CHECK: %[[BEFORE:.*]] = ondrix.fir
// CHECK: memref.store
// CHECK: %[[AFTER:.*]] = ondrix.fir
// CHECK: return %[[BEFORE]], %[[AFTER]]

func.func @fir_reads_second_operand(%input: f32, %coeffs: memref<8xf32>, %value: f32) -> (f32, f32) {
  %c0 = arith.constant 0 : index
  %before = ondrix.fir %input, %coeffs {numeric = #ondsp.fp<format = f32, contract = off>} : (f32, memref<8xf32>) -> f32
  memref.store %value, %coeffs[%c0] : memref<8xf32>
  %after = ondrix.fir %input, %coeffs {numeric = #ondsp.fp<format = f32, contract = off>} : (f32, memref<8xf32>) -> f32
  return %before, %after : f32, f32
}

// CHECK-LABEL: func.func @fir_reads_second_operand
// CHECK: %[[BEFORE:.*]] = ondrix.fir
// CHECK: memref.store
// CHECK: %[[AFTER:.*]] = ondrix.fir
// CHECK: return %[[BEFORE]], %[[AFTER]]

func.func @dot_reads_memory(%lhs: memref<8xf32>, %rhs: memref<8xf32>, %value: f32) -> (f32, f32) {
  %c0 = arith.constant 0 : index
  %before = ondrix.dot %lhs, %rhs {numeric = #ondsp.fp<format = f32, contract = off>} : (memref<8xf32>, memref<8xf32>) -> f32
  memref.store %value, %lhs[%c0] : memref<8xf32>
  %after = ondrix.dot %lhs, %rhs {numeric = #ondsp.fp<format = f32, contract = off>} : (memref<8xf32>, memref<8xf32>) -> f32
  return %before, %after : f32, f32
}

// CHECK-LABEL: func.func @dot_reads_memory
// CHECK: %[[BEFORE:.*]] = ondrix.dot
// CHECK: memref.store
// CHECK: %[[AFTER:.*]] = ondrix.dot
// CHECK: return %[[BEFORE]], %[[AFTER]]

func.func @dot_reads_unranked_first_operand(%lhs: memref<8xf32>, %rhs: f32, %value: f32) -> (f32, f32) {
  %c0 = arith.constant 0 : index
  %unranked = memref.cast %lhs : memref<8xf32> to memref<*xf32>
  %before = ondrix.dot %unranked, %rhs {numeric = #ondsp.fp<format = f32, contract = off>} : (memref<*xf32>, f32) -> f32
  memref.store %value, %lhs[%c0] : memref<8xf32>
  %after = ondrix.dot %unranked, %rhs {numeric = #ondsp.fp<format = f32, contract = off>} : (memref<*xf32>, f32) -> f32
  return %before, %after : f32, f32
}

// CHECK-LABEL: func.func @dot_reads_unranked_first_operand
// CHECK: %[[BEFORE:.*]] = ondrix.dot
// CHECK: memref.store
// CHECK: %[[AFTER:.*]] = ondrix.dot
// CHECK: return %[[BEFORE]], %[[AFTER]]

func.func @dot_reads_unranked_second_operand(%lhs: f32, %rhs: memref<8xf32>, %value: f32) -> (f32, f32) {
  %c0 = arith.constant 0 : index
  %unranked = memref.cast %rhs : memref<8xf32> to memref<*xf32>
  %before = ondrix.dot %lhs, %unranked {numeric = #ondsp.fp<format = f32, contract = off>} : (f32, memref<*xf32>) -> f32
  memref.store %value, %rhs[%c0] : memref<8xf32>
  %after = ondrix.dot %lhs, %unranked {numeric = #ondsp.fp<format = f32, contract = off>} : (f32, memref<*xf32>) -> f32
  return %before, %after : f32, f32
}

// CHECK-LABEL: func.func @dot_reads_unranked_second_operand
// CHECK: %[[BEFORE:.*]] = ondrix.dot
// CHECK: memref.store
// CHECK: %[[AFTER:.*]] = ondrix.dot
// CHECK: return %[[BEFORE]], %[[AFTER]]

func.func @scalar_dot_is_memory_effect_free(%lhs: i16, %rhs: i16) -> (i32, i32) {
  %first = ondrix.dot %lhs, %rhs {numeric = #ondsp.fixed<signed, storage = i16, frac = 15>, product = #ondsp.product<full>} : (i16, i16) -> i32
  %second = ondrix.dot %lhs, %rhs {numeric = #ondsp.fixed<signed, storage = i16, frac = 15>, product = #ondsp.product<full>} : (i16, i16) -> i32
  return %first, %second : i32, i32
}

// CHECK-LABEL: func.func @scalar_dot_is_memory_effect_free
// CHECK: %[[DOT:.*]] = ondrix.dot
// CHECK-NOT: ondrix.dot
// CHECK: return %[[DOT]], %[[DOT]]

func.func @scalar_value_ops_are_speculatable(%lhs: i16, %rhs: i16, %upper: index) -> i32 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %init = arith.constant 0 : i32
  %result = scf.for %i = %c0 to %upper step %c1 iter_args(%acc = %init) -> (i32) {
    %fir = ondrix.fir %lhs, %rhs {numeric = #ondsp.fixed<signed, storage = i16, frac = 15>, product = #ondsp.product<full>} : (i16, i16) -> i32
    %dot = ondrix.dot %lhs, %rhs {numeric = #ondsp.fixed<signed, storage = i16, frac = 15>, product = #ondsp.product<full>} : (i16, i16) -> i32
    %sum = arith.addi %fir, %dot : i32
    %next = arith.addi %acc, %sum : i32
    scf.yield %next : i32
  }
  return %result : i32
}

// LICM-LABEL: func.func @scalar_value_ops_are_speculatable
// LICM: %[[FIR:.*]] = ondrix.fir
// LICM: %[[DOT:.*]] = ondrix.dot
// LICM: scf.for
// LICM-NOT: ondrix.fir
// LICM-NOT: ondrix.dot

func.func @unranked_buffer_ops_remain_in_loop(%lhs: memref<*xf32>, %rhs: memref<*xf32>, %upper: index) -> f32 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %init = arith.constant 0.0 : f32
  %result = scf.for %i = %c0 to %upper step %c1 iter_args(%acc = %init) -> (f32) {
    %fir = ondrix.fir %lhs, %rhs {numeric = #ondsp.fp<format = f32, contract = off>} : (memref<*xf32>, memref<*xf32>) -> f32
    %dot = ondrix.dot %lhs, %rhs {numeric = #ondsp.fp<format = f32, contract = off>} : (memref<*xf32>, memref<*xf32>) -> f32
    %sum = arith.addf %fir, %dot : f32
    %next = arith.addf %acc, %sum : f32
    scf.yield %next : f32
  }
  return %result : f32
}

// LICM-LABEL: func.func @unranked_buffer_ops_remain_in_loop
// LICM: scf.for
// LICM: ondrix.fir
// LICM: ondrix.dot
