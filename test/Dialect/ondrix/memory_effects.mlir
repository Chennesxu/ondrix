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

func.func @fir_reads_coefficients(%input: memref<8xf32>, %coeffs: memref<8xf32>, %value: f32) -> (f32, f32) {
  %c0 = arith.constant 0 : index
  %before = ondrix.fir %input, %coeffs {numeric = #ondsp.fp<format = f32, contract = off>} : (memref<8xf32>, memref<8xf32>) -> f32
  memref.store %value, %coeffs[%c0] : memref<8xf32>
  %after = ondrix.fir %input, %coeffs {numeric = #ondsp.fp<format = f32, contract = off>} : (memref<8xf32>, memref<8xf32>) -> f32
  return %before, %after : f32, f32
}

// CHECK-LABEL: func.func @fir_reads_coefficients
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

func.func @scalar_dot_is_memory_effect_free(%lhs: i16, %rhs: i16) -> (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>, !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>) {
  %first = ondrix.dot %lhs, %rhs {numeric = #ondsp.fixed<signed, storage = i16, frac = 15>, product = #ondsp.product<full>} : (i16, i16) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  %second = ondrix.dot %lhs, %rhs {numeric = #ondsp.fixed<signed, storage = i16, frac = 15>, product = #ondsp.product<full>} : (i16, i16) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  return %first, %second : !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>, !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
}

// CHECK-LABEL: func.func @scalar_dot_is_memory_effect_free
// CHECK: %[[DOT:.*]] = ondrix.dot
// CHECK-NOT: ondrix.dot
// CHECK: return %[[DOT]], %[[DOT]]

func.func @scalar_dot_is_speculatable(%lhs: i16, %rhs: i16, %upper: index) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %init = ondsp.acc_zero : !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  %result = scf.for %i = %c0 to %upper step %c1 iter_args(%acc = %init) -> (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>) {
    %dot = ondrix.dot %lhs, %rhs {numeric = #ondsp.fixed<signed, storage = i16, frac = 15>, product = #ondsp.product<full>} : (i16, i16) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
    scf.yield %dot : !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  }
  return %result : !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
}

// LICM-LABEL: func.func @scalar_dot_is_speculatable
// LICM: %[[DOT:.*]] = ondrix.dot
// LICM: scf.for
// LICM-NOT: ondrix.dot

func.func @static_tensor_fir_filter_is_speculatable(
    %input: tensor<8xf32>, %coeffs: tensor<3xf32>, %init: tensor<6xf32>,
    %upper: index) -> tensor<6xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %result = scf.for %i = %c0 to %upper step %c1
      iter_args(%current = %init) -> tensor<6xf32> {
    %filtered = ondrix.fir_filter %input, %coeffs, %init {
      boundary = #ondrix.fir_boundary<valid>,
      numeric = #ondsp.fp<format = f32, contract = off>
    } : (tensor<8xf32>, tensor<3xf32>, tensor<6xf32>) -> tensor<6xf32>
    scf.yield %filtered : tensor<6xf32>
  }
  return %result : tensor<6xf32>
}

// LICM-LABEL: func.func @static_tensor_fir_filter_is_speculatable
// LICM: %[[FILTER:.*]] = ondrix.fir_filter
// LICM: scf.for
// LICM-NOT: ondrix.fir_filter

func.func @dynamic_tensor_fir_filter_remains_in_loop(
    %input: tensor<?xf32>, %coeffs: tensor<?xf32>, %init: tensor<?xf32>,
    %upper: index) -> tensor<?xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %result = scf.for %i = %c0 to %upper step %c1
      iter_args(%current = %init) -> tensor<?xf32> {
    %filtered = ondrix.fir_filter %input, %coeffs, %init {
      boundary = #ondrix.fir_boundary<valid>,
      numeric = #ondsp.fp<format = f32, contract = off>
    } : (tensor<?xf32>, tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
    scf.yield %filtered : tensor<?xf32>
  }
  return %result : tensor<?xf32>
}

// LICM-LABEL: func.func @dynamic_tensor_fir_filter_remains_in_loop
// LICM: scf.for
// LICM: ondrix.fir_filter

func.func @origin_fir_filter_remains_in_loop(
    %input: tensor<4xf32>, %coeffs: tensor<3xf32>, %init: tensor<1xf32>,
    %upper: index) -> tensor<1xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %origin = arith.constant 2 : index
  %result = scf.for %i = %c0 to %upper step %c1
      iter_args(%current = %init) -> tensor<1xf32> {
    %filtered = ondrix.fir_filter %input, %coeffs, %init, %origin {
      boundary = #ondrix.fir_boundary<full>,
      numeric = #ondsp.fp<format = f32, contract = off>
    } : (tensor<4xf32>, tensor<3xf32>, tensor<1xf32>, index) -> tensor<1xf32>
    scf.yield %filtered : tensor<1xf32>
  }
  return %result : tensor<1xf32>
}

// LICM-LABEL: func.func @origin_fir_filter_remains_in_loop
// LICM: scf.for
// LICM: ondrix.fir_filter
