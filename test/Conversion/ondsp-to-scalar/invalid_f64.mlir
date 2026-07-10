// RUN: not ondrix-opt %s --convert-ondsp-to-scalar 2>&1 | FileCheck %s

func.func @f64(%lhs: memref<8xf64>, %rhs: memref<8xf64>) -> f64 {
  // CHECK: scalar lowering requires numeric = #ondsp.fp<format = f32, ...>
  %0 = ondsp.reduce_mac %lhs, %rhs {numeric = #ondsp.fp<format = f64, contract = fma>} : (memref<8xf64>, memref<8xf64>) -> f64
  return %0 : f64
}
