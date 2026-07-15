// RUN: ondrix-opt %s --cse | FileCheck %s
// RUN: ondrix-opt %s --loop-invariant-code-motion | FileCheck %s --check-prefix=LICM

func.func @reduce_mac_reads_memory(%lhs: memref<8xf32>, %rhs: memref<8xf32>, %value: f32) -> (f32, f32) {
  %c0 = arith.constant 0 : index
  %zero = arith.constant 0.0 : f32
  %before = ondsp.reduce_mac %zero, %lhs, %rhs {numeric = #ondsp.fp<format = f32, contract = off>} : (f32, memref<8xf32>, memref<8xf32>) -> f32
  memref.store %value, %lhs[%c0] : memref<8xf32>
  %after = ondsp.reduce_mac %zero, %lhs, %rhs {numeric = #ondsp.fp<format = f32, contract = off>} : (f32, memref<8xf32>, memref<8xf32>) -> f32
  return %before, %after : f32, f32
}

// CHECK-LABEL: func.func @reduce_mac_reads_memory
// CHECK: %[[BEFORE:.*]] = ondsp.reduce_mac
// CHECK: memref.store
// CHECK: %[[AFTER:.*]] = ondsp.reduce_mac
// CHECK: return %[[BEFORE]], %[[AFTER]]

func.func @value_reduce_mac_is_memory_effect_free(%lhs: vector<8xi16>, %rhs: vector<8xi16>) -> (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>, !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>) {
  %zero = ondsp.acc_zero : !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  %first = ondsp.reduce_mac %zero, %lhs, %rhs {numeric = #ondsp.fixed<signed, storage = i16, frac = 15>, product = #ondsp.product<full>} : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>, vector<8xi16>, vector<8xi16>) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  %second = ondsp.reduce_mac %zero, %lhs, %rhs {numeric = #ondsp.fixed<signed, storage = i16, frac = 15>, product = #ondsp.product<full>} : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>, vector<8xi16>, vector<8xi16>) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  return %first, %second : !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>, !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
}

// CHECK-LABEL: func.func @value_reduce_mac_is_memory_effect_free
// CHECK: %[[REDUCE:.*]] = ondsp.reduce_mac
// CHECK-NOT: ondsp.reduce_mac
// CHECK: return %[[REDUCE]], %[[REDUCE]]

func.func @value_reduce_mac_is_speculatable(%lhs: vector<8xi16>, %rhs: vector<8xi16>, %upper: index) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %init = ondsp.acc_zero : !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  %result = scf.for %i = %c0 to %upper step %c1 iter_args(%acc = %init) -> (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>) {
    %reduce = ondsp.reduce_mac %init, %lhs, %rhs {numeric = #ondsp.fixed<signed, storage = i16, frac = 15>, product = #ondsp.product<full>} : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>, vector<8xi16>, vector<8xi16>) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
    scf.yield %reduce : !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  }
  return %result : !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
}

// LICM-LABEL: func.func @value_reduce_mac_is_speculatable
// LICM: %[[REDUCE:.*]] = ondsp.reduce_mac
// LICM: scf.for
// LICM-NOT: ondsp.reduce_mac

func.func @static_vector_reduce_mac_is_speculatable(
    %lhs: vector<8xf32>, %rhs: vector<8xf32>, %upper: index) -> f32 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %init = arith.constant 0.0 : f32
  %result = scf.for %i = %c0 to %upper step %c1 iter_args(%acc = %init) -> (f32) {
    %reduce = ondsp.reduce_mac %init, %lhs, %rhs {numeric = #ondsp.fp<format = f32, contract = off>} : (f32, vector<8xf32>, vector<8xf32>) -> f32
    %next = arith.addf %acc, %reduce : f32
    scf.yield %next : f32
  }
  return %result : f32
}

// LICM-LABEL: func.func @static_vector_reduce_mac_is_speculatable
// LICM: %[[REDUCE:.*]] = ondsp.reduce_mac
// LICM: scf.for
// LICM-NOT: ondsp.reduce_mac

func.func @dynamic_tensor_reduce_mac_remains_in_loop(
    %lhs: tensor<?xf32>, %rhs: tensor<?xf32>, %upper: index) -> f32 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %init = arith.constant 0.0 : f32
  %result = scf.for %i = %c0 to %upper step %c1 iter_args(%acc = %init) -> (f32) {
    %reduce = ondsp.reduce_mac %init, %lhs, %rhs {numeric = #ondsp.fp<format = f32, contract = off>} : (f32, tensor<?xf32>, tensor<?xf32>) -> f32
    %next = arith.addf %acc, %reduce : f32
    scf.yield %next : f32
  }
  return %result : f32
}

// LICM-LABEL: func.func @dynamic_tensor_reduce_mac_remains_in_loop
// LICM: scf.for
// LICM: ondsp.reduce_mac
