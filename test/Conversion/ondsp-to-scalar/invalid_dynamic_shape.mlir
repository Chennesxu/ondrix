// RUN: not ondrix-opt %s --convert-ondsp-to-scalar 2>&1 | FileCheck %s

func.func @dynamic_shape(%lhs: memref<?xf32>, %rhs: memref<?xf32>) -> f32 {
  // CHECK: lhs must have a static shape for scalar lowering
  %0 = ondsp.reduce_mac %lhs, %rhs {numeric = #ondsp.fp<format = f32, contract = fma>} : (memref<?xf32>, memref<?xf32>) -> f32
  return %0 : f32
}
