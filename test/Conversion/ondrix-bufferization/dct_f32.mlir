// RUN: ondrix-opt %s --one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map allow-return-allocs" | FileCheck %s
// RUN: ondrix-opt %s --one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map allow-return-allocs" --canonicalize --lower-ondsp-f32-reduce-to-scalar="vector-width=4" | FileCheck %s --check-prefix=BATCHED

// The binary32 profile derives its table in binary32 with no tie guard: the
// guard certifies a quantized table against an independently specified value,
// and here the binary32 rounding IS the declared constant.
// CHECK-DAG: memref.global "private" constant @__ondrix_dct8_f32_row0 : memref<8xf32> = dense<1.000000e+00>
// CHECK-DAG: memref.global "private" constant @__ondrix_dct8_f32_row4 : memref<8xf32> = dense<[0.707106769, -0.707106769, -0.707106769, 0.707106769, 0.707106769, -0.707106769, -0.707106769, 0.707106769]>

// CHECK-LABEL: func.func @f32_dct8
// CHECK: %[[OUT:.*]] = memref.alloc() {alignment = 64 : i64} : memref<8xf32>
// CHECK: %[[C0:.*]] = arith.constant 0 : index
// CHECK: %[[ROW:.*]] = memref.get_global @__ondrix_dct8_f32_row0
// Each row starts AT its first product; seeding the additive identity instead
// would export +0.0 where the contract exports -0.0.
// CHECK: %[[X0:.*]] = memref.load %{{.*}}[%[[C0]]] : memref<8xf32>
// CHECK: %[[C00:.*]] = memref.load %[[ROW]][%[[C0]]] : memref<8xf32>
// CHECK: %[[SEED:.*]] = arith.mulf %[[X0]], %[[C00]] : f32
// CHECK: %[[XT:.*]] = memref.subview %{{.*}}[1] [7] [1] : memref<8xf32> to memref<7xf32, strided<[1], offset: 1>>
// CHECK: %[[CT:.*]] = memref.subview %[[ROW]][1] [7] [1] : memref<8xf32> to memref<7xf32, strided<[1], offset: 1>>
// CHECK: %[[SUM:.*]] = ondsp.reduce_mac %[[SEED]], %[[XT]], %[[CT]]
// CHECK: memref.store %[[SUM]], %[[OUT]][%{{.*}}]

// The reduction is a loop over a unit-stride view, so the declared-off row
// reaches the ordered product batching the straight-line form could not.
// BATCHED-LABEL: func.func @f32_dct8
// BATCHED: vector.load {{.*}} : memref<7xf32, strided<[1], offset: 1>>, vector<4xf32>
// BATCHED: arith.mulf {{.*}} : vector<4xf32>

func.func @f32_dct8(%input: tensor<8xf32>) -> tensor<8xf32> {
  %result = ondrix.dct %input {
    input_numeric = #ondsp.fp<format = f32, contract = off>,
    output_numeric = #ondsp.fp<format = f32, contract = off>
  } : (tensor<8xf32>) -> tensor<8xf32>
  return %result : tensor<8xf32>
}

// A fused row keeps the same seed and the same tail view; only the contract
// the reduction carries differs.
// CHECK-LABEL: func.func @f32_dct32_fast
// CHECK: arith.mulf
// CHECK: ondsp.reduce_mac {{.*}} contract = fast
// CHECK-NOT: ondrix.dct

func.func @f32_dct32_fast(%input: tensor<32xf32>) -> tensor<32xf32> {
  %result = ondrix.dct %input {
    input_numeric = #ondsp.fp<format = f32, contract = fast>,
    output_numeric = #ondsp.fp<format = f32, contract = fast>
  } : (tensor<32xf32>) -> tensor<32xf32>
  return %result : tensor<32xf32>
}
