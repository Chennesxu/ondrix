// RUN: not ondrix-opt %s --convert-ondsp-to-scalar 2>&1 | FileCheck %s

func.func @shape_mismatch(%lhs: memref<8xf32>, %rhs: memref<4xf32>) -> f32 {
  // CHECK: scalar lowering requires lhs and rhs to have equal static lengths
  %0 = ondsp.reduce_mac %lhs, %rhs {numeric = #ondsp.fp<format = f32, contract = fma>} : (memref<8xf32>, memref<4xf32>) -> f32
  return %0 : f32
}
