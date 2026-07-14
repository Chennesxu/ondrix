// RUN: ondrix-opt %s --lower-ondsp-f32-reduce-to-scalar | FileCheck %s

func.func @preserve_f64(%lhs: memref<8xf64>, %rhs: memref<8xf64>) -> f64 {
  %0 = ondsp.reduce_mac %lhs, %rhs {numeric = #ondsp.fp<format = f64, contract = fma>} : (memref<8xf64>, memref<8xf64>) -> f64
  return %0 : f64
}

// CHECK-LABEL: func.func @preserve_f64
// CHECK: ondsp.reduce_mac

func.func @preserve_vector(%lhs: vector<8xf32>, %rhs: vector<8xf32>) -> f32 {
  %0 = ondsp.reduce_mac %lhs, %rhs {numeric = #ondsp.fp<format = f32, contract = fma>} : (vector<8xf32>, vector<8xf32>) -> f32
  return %0 : f32
}

// CHECK-LABEL: func.func @preserve_vector
// CHECK: ondsp.reduce_mac
