// RUN: ondrix-opt %s --one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map" | FileCheck %s

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
