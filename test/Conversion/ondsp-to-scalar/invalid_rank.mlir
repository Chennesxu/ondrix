// RUN: not ondrix-opt %s --convert-ondsp-to-scalar 2>&1 | FileCheck %s

func.func @rank_two(%lhs: memref<2x4xf32>, %rhs: memref<2x4xf32>) -> f32 {
  // CHECK: lhs must be a rank-1 memref<Nxf32> for scalar lowering
  %0 = ondsp.reduce_mac %lhs, %rhs {numeric = #ondsp.fp<format = f32, contract = fma>} : (memref<2x4xf32>, memref<2x4xf32>) -> f32
  return %0 : f32
}
