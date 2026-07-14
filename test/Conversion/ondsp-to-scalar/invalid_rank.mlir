// RUN: not ondrix-opt %s --lower-ondsp-f32-reduce-to-scalar 2>&1 | FileCheck %s

func.func @rank_two(%lhs: memref<2x4xf32>, %rhs: memref<2x4xf32>) -> f32 {
  // CHECK: shaped operands must be rank-1
  %0 = ondsp.reduce_mac %lhs, %rhs {numeric = #ondsp.fp<format = f32, contract = fma>} : (memref<2x4xf32>, memref<2x4xf32>) -> f32
  return %0 : f32
}
