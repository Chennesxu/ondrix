// RUN: ondrix-opt %s --lower-ondsp-f32-reduce-to-scalar | FileCheck %s

func.func @preserve_f64(%lhs: memref<8xf64>, %rhs: memref<8xf64>) -> f64 {
  %zero = arith.constant 0.0 : f64
  %0 = ondsp.reduce_mac %zero, %lhs, %rhs {numeric = #ondsp.fp<format = f64, contract = fma>} : (f64, memref<8xf64>, memref<8xf64>) -> f64
  return %0 : f64
}

// CHECK-LABEL: func.func @preserve_f64
// CHECK: ondsp.reduce_mac
