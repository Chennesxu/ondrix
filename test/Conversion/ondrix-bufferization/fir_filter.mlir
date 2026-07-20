// RUN: ondrix-opt %s --one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map" | FileCheck %s
// RUN: ondrix-opt %s --one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map" --canonicalize --vectorize-ondsp-fixed-memref-reduce="vector-width=4" --normalize-ondsp-fixed-vector-reduce | FileCheck %s --check-prefix=FULL-VECTOR

// CHECK-LABEL: func.func @q15(
// CHECK-SAME: %[[INPUT:.*]]: memref<?xi16>, %[[COEFFS:.*]]: memref<?xi16>, %[[OUTPUT:.*]]: memref<?xi16>)
// CHECK-NOT: cf.assert
// CHECK-NOT: ondrix.fir_filter
// CHECK-NOT: memref.alloc
// CHECK-NOT: memref.copy
// CHECK: %[[COEFF_LENGTH:.*]] = memref.dim %[[COEFFS]], %{{.*}}
// CHECK: %[[COEFF_VIEW:.*]] = memref.subview %[[COEFFS]][0] [%[[COEFF_LENGTH]]] [1]
// CHECK: scf.for %[[OUTPUT_INDEX:.*]] =
// CHECK-NOT: cf.assert
// CHECK: %[[WINDOW:.*]] = memref.subview %[[INPUT]][%[[OUTPUT_INDEX]]] [%[[COEFF_LENGTH]]] [1]
// CHECK: %[[INITIAL:.*]] = ondsp.acc_zero
// CHECK: %[[REDUCED:.*]] = ondsp.reduce_mac %[[INITIAL]], %[[WINDOW]], %[[COEFF_VIEW]]
// CHECK: %[[SAMPLE:.*]] = ondsp.acc_export %[[REDUCED]]
// CHECK: memref.store %[[SAMPLE]], %[[OUTPUT]][%[[OUTPUT_INDEX]]]
// CHECK-NOT: cf.assert
// CHECK-NOT: ondrix.fir_filter
// CHECK-NOT: memref.alloc
// CHECK-NOT: memref.copy
// CHECK: return %[[OUTPUT]]

func.func @q15(
    %input: tensor<?xi16>, %coeffs: tensor<?xi16>, %init: tensor<?xi16>)
    -> tensor<?xi16> {
  %result = ondrix.fir_filter %input, %coeffs, %init {
    accumulator = !ondsp.acc<storage = i40, frac = 30, signed,
                              update_overflow = saturate>,
    boundary = #ondrix.fir_boundary<valid>,
    dst = #ondsp.fixed<signed, storage = i16, frac = 15>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    overflow = #ondsp.overflow<saturate>,
    product = #ondsp.product<full>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<?xi16>, tensor<?xi16>, tensor<?xi16>) -> tensor<?xi16>
  return %result : tensor<?xi16>
}

// CHECK-LABEL: func.func @q15_full(
// CHECK-SAME: %[[FULL_INPUT:.*]]: memref<?xi16>, %[[FULL_COEFFS:.*]]: memref<?xi16>, %[[FULL_OUTPUT:.*]]: memref<?xi16>)
// CHECK-COUNT-3: cf.assert
// CHECK: scf.for
// CHECK: scf.for
// CHECK: scf.if
// CHECK: ondsp.mac
// CHECK: memref.store {{.*}}, %[[FULL_OUTPUT]]
// CHECK: scf.for
// CHECK: memref.subview %[[FULL_INPUT]]
// CHECK: ondsp.reduce_mac
// CHECK: memref.store {{.*}}, %[[FULL_OUTPUT]]
// CHECK: arith.select
// CHECK: scf.for
// CHECK: scf.if
// CHECK: ondsp.mac
// CHECK: memref.store {{.*}}, %[[FULL_OUTPUT]]
// CHECK-NOT: ondrix.fir_filter
// CHECK: return %[[FULL_OUTPUT]]
func.func @q15_full(
    %input: tensor<?xi16>, %coeffs: tensor<?xi16>, %init: tensor<?xi16>)
    -> tensor<?xi16> {
  %result = ondrix.fir_filter %input, %coeffs, %init {
    accumulator = !ondsp.acc<storage = i40, frac = 30, signed,
                              update_overflow = saturate>,
    boundary = #ondrix.fir_boundary<full>,
    dst = #ondsp.fixed<signed, storage = i16, frac = 15>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    overflow = #ondsp.overflow<saturate>,
    product = #ondsp.product<full>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<?xi16>, tensor<?xi16>, tensor<?xi16>) -> tensor<?xi16>
  return %result : tensor<?xi16>
}

// FULL-VECTOR-LABEL: func.func @q15_full
// FULL-VECTOR: scf.if
// FULL-VECTOR: ondsp.mac
// FULL-VECTOR-COUNT-2: vector.load {{.*}}vector<4xi16>
// FULL-VECTOR: arith.muli {{.*}} : vector<4xi32>
// FULL-VECTOR: scf.if
// FULL-VECTOR: ondsp.mac
// FULL-VECTOR-NOT: ondsp.reduce_mac
