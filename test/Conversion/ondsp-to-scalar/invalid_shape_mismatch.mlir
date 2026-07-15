// RUN: not ondrix-opt %s --lower-ondsp-f32-reduce-to-scalar 2>&1 | FileCheck %s

func.func @shape_mismatch(%lhs: memref<8xf32>, %rhs: memref<4xf32>) -> f32 {
  %zero = arith.constant 0.0 : f32
  // CHECK: shaped operands must have equal static lengths
  %0 = ondsp.reduce_mac %zero, %lhs, %rhs {numeric = #ondsp.fp<format = f32, contract = fma>} : (f32, memref<8xf32>, memref<4xf32>) -> f32
  return %0 : f32
}
