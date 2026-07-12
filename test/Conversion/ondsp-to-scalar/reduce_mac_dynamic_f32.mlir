// RUN: ondrix-opt %s --convert-ondsp-to-scalar | FileCheck %s

func.func @reduce_mac_dynamic(%lhs: memref<?xf32>, %rhs: memref<?xf32>) -> f32 {
  %0 = ondsp.reduce_mac %lhs, %rhs {numeric = #ondsp.fp<format = f32, contract = fma>} : (memref<?xf32>, memref<?xf32>) -> f32
  return %0 : f32
}

// CHECK-LABEL: func.func @reduce_mac_dynamic
// CHECK: %[[C0:.*]] = arith.constant 0 : index
// CHECK: %[[LHS_LEN:.*]] = memref.dim %arg0, %[[C0]] : memref<?xf32>
// CHECK: %[[RHS_LEN:.*]] = memref.dim %arg1, %[[C0]] : memref<?xf32>
// CHECK: %[[MATCH:.*]] = arith.cmpi eq, %[[LHS_LEN]], %[[RHS_LEN]] : index
// CHECK: cf.assert %[[MATCH]], "ondsp.reduce_mac requires equal operand lengths"
// CHECK: scf.for {{.*}} to %[[LHS_LEN]] {{.*}} iter_args
// CHECK: math.fma
// CHECK-NOT: ondsp.reduce_mac
