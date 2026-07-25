// RUN: ondrix-opt %s --one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map" | FileCheck %s
// RUN: ondrix-opt %s --one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map" --cse --canonicalize --vectorize-ondsp-fixed-memref-reduce="vector-width=4" --normalize-ondsp-fixed-vector-reduce | FileCheck %s --check-prefix=VECTOR

// CHECK-LABEL: func.func @q15_decimate(
// CHECK-SAME: %[[INPUT:.*]]: memref<?xi16>, %[[COEFFS:.*]]: memref<?xi16>, %[[OUTPUT:.*]]: memref<?xi16>)
// CHECK: %[[FACTOR:.*]] = arith.constant 2 : index
// CHECK: %[[INPUT_LENGTH:.*]] = memref.dim %[[INPUT]]
// CHECK: %[[COEFF_LENGTH:.*]] = memref.dim %[[COEFFS]]
// CHECK: %[[OUTPUT_LENGTH:.*]] = memref.dim %[[OUTPUT]]
// CHECK-COUNT-3: cf.assert
// CHECK: %[[COEFF_VIEW:.*]] = memref.subview %[[COEFFS]][0] [%[[COEFF_LENGTH]]] [1]
// CHECK: scf.for %[[OUTPUT_INDEX:.*]] =
// CHECK: %[[INPUT_OFFSET:.*]] = arith.muli %[[OUTPUT_INDEX]], %[[FACTOR]] : index
// CHECK: %[[WINDOW:.*]] = memref.subview %[[INPUT]][%[[INPUT_OFFSET]]] [%[[COEFF_LENGTH]]] [1]
// CHECK: %[[INITIAL:.*]] = ondsp.acc_zero
// CHECK: %[[REDUCED:.*]] = ondsp.reduce_mac %[[INITIAL]], %[[WINDOW]], %[[COEFF_VIEW]]
// CHECK: %[[SAMPLE:.*]] = ondsp.acc_export %[[REDUCED]]
// CHECK: memref.store %[[SAMPLE]], %[[OUTPUT]][%[[OUTPUT_INDEX]]]
// CHECK-NOT: ondrix.fir_decimate
// CHECK: return %[[OUTPUT]]

// VECTOR-LABEL: func.func @q15_decimate
// VECTOR: %[[VECTOR_FACTOR:.*]] = arith.constant 2 : index
// VECTOR: arith.muli {{.*}}, %[[VECTOR_FACTOR]] : index
// VECTOR-COUNT-2: vector.load {{.*}}vector<4xi16>
// VECTOR: arith.muli {{.*}} : vector<4xi32>
// VECTOR-COUNT-4: ondsp.acc_add_term
// VECTOR: ondsp.mac
// VECTOR-NOT: ondsp.reduce_mac
// VECTOR-NOT: ondrix.fir_decimate
func.func @q15_decimate(
    %input: tensor<?xi16>, %coeffs: tensor<?xi16>, %init: tensor<?xi16>)
    -> tensor<?xi16> {
  %result = ondrix.fir_decimate %input, %coeffs, %init {
    accumulator = !ondsp.acc<storage = i34, frac = 30, signed,
                              update_overflow = wrap>,
    dst = #ondsp.fixed<signed, storage = i16, frac = 15>,
    factor = 2,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    overflow = #ondsp.overflow<saturate>,
    product = #ondsp.product<full>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<?xi16>, tensor<?xi16>, tensor<?xi16>) -> tensor<?xi16>
  return %result : tensor<?xi16>
}
