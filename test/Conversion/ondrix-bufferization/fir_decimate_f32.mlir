// RUN: ondrix-opt %s --one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map" | FileCheck %s
// RUN: ondrix-opt %s --one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map" --cse --canonicalize --vectorize-ondsp-fp-fast-memref-reduce="vector-width=8" | FileCheck %s --check-prefix=VECTOR

// The strided input window a decimation reads is still unit stride, so the
// f32 profile reaches the same memref reduction the FIR filter does and the
// fast contract admits the batched schedule.

// CHECK-LABEL: func.func @f32_decimate(
// CHECK: %[[FACTOR:.*]] = arith.constant 2 : index
// CHECK: %[[COEFF_VIEW:.*]] = memref.subview
// CHECK: scf.for %[[OUTPUT_INDEX:.*]] =
// CHECK: %[[INPUT_OFFSET:.*]] = arith.muli %[[OUTPUT_INDEX]], %[[FACTOR]] : index
// CHECK: %[[WINDOW:.*]] = memref.subview {{.*}}[%[[INPUT_OFFSET]]]
// CHECK: %[[SEED:.*]] = arith.constant 0.000000e+00 : f32
// CHECK: %[[REDUCED:.*]] = ondsp.reduce_mac %[[SEED]], %[[WINDOW]], %[[COEFF_VIEW]]
// CHECK-SAME: numeric = #ondsp.fp<format = f32, contract = fast>
// CHECK: memref.store %[[REDUCED]]
// CHECK-NOT: ondsp.acc_export
// CHECK-NOT: ondrix.fir_decimate

// VECTOR-LABEL: func.func @f32_decimate
// VECTOR-COUNT-2: vector.load {{.*}}vector<8xf32>
// VECTOR: arith.mulf %{{[^ ]*}}, %{{[^ ]*}} : vector<8xf32>
// VECTOR: vector.shuffle {{.*}} [0, 1, 2, 3] : vector<8xf32>, vector<8xf32>
// VECTOR: arith.addf {{.*}} {ondsp.fast_used = ["rebuild_reduction_tree"]} : f32
// VECTOR-NOT: ondsp.reduce_mac
func.func @f32_decimate(
    %input: tensor<?xf32>, %coeffs: tensor<?xf32>, %init: tensor<?xf32>)
    -> tensor<?xf32> {
  %result = ondrix.fir_decimate %input, %coeffs, %init {
    factor = 2,
    numeric = #ondsp.fp<format = f32, contract = fast>
  } : (tensor<?xf32>, tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  return %result : tensor<?xf32>
}
