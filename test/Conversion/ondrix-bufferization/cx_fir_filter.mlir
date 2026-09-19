// RUN: ondrix-opt %s --one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map" | FileCheck %s

!acc = !ondsp.acc<storage = i32, frac = 30, signed, update_overflow = saturate>

// The window is a view rather than a copy, so the whole filter is one loop
// whose body is the packed complex reduction the target route selects.
func.func @cx_fir_valid(%input: tensor<?xi32>, %coeffs: tensor<?xi32>,
                        %init: tensor<?xi32>) -> tensor<?xi32> {
  %out = ondrix.cx_fir_filter %input, %coeffs, %init {
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    layout = #ondsp.cx_layout<packed_i16_imag_hi_real_lo>,
    accumulator = !acc,
    rounding = #ondsp.rounding<toward_negative>,
    overflow = #ondsp.overflow<saturate>
  } : (tensor<?xi32>, tensor<?xi32>, tensor<?xi32>) -> tensor<?xi32>
  return %out : tensor<?xi32>
}

// CHECK-LABEL: func.func @cx_fir_valid(
// CHECK-SAME: %[[INPUT:.*]]: memref<?xi32>, %[[COEFFS:.*]]: memref<?xi32>, %[[OUTPUT:.*]]: memref<?xi32>)
// CHECK-NOT: memref.alloc
// CHECK: %[[TAPS:.*]] = memref.dim %[[COEFFS]], %{{.*}}
// CHECK: %[[TAP_VIEW:.*]] = memref.subview %[[COEFFS]][0] [%[[TAPS]]] [1]
// CHECK: scf.for %[[N:.*]] =
// CHECK: %[[WINDOW:.*]] = memref.subview %[[INPUT]][%[[N]]] [%[[TAPS]]] [1]
// CHECK: %[[RE:.*]] = ondsp.acc_zero
// CHECK: %[[IM:.*]] = ondsp.acc_zero
// CHECK: ondsp.cx_reduce_mac %[[RE]], %[[IM]], %[[WINDOW]], %[[TAP_VIEW]]
// CHECK-NOT: conjugate
// CHECK: ondsp.acc_export
// CHECK: ondsp.acc_export
// CHECK: arith.extui
// CHECK: arith.extui
// CHECK: arith.shli
// CHECK: arith.ori
// CHECK: memref.store %{{.*}}, %[[OUTPUT]][%[[N]]]
// CHECK: return %[[OUTPUT]]

// The correlation differs from the filter by the flag alone, which the
// reduction carries down unchanged.
func.func @cx_correlate_valid(%input: tensor<?xi32>, %coeffs: tensor<?xi32>,
                              %init: tensor<?xi32>) -> tensor<?xi32> {
  %out = ondrix.cx_fir_filter %input, %coeffs, %init {
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    layout = #ondsp.cx_layout<packed_i16_imag_hi_real_lo>,
    accumulator = !acc,
    rounding = #ondsp.rounding<toward_negative>,
    overflow = #ondsp.overflow<saturate>,
    conjugate
  } : (tensor<?xi32>, tensor<?xi32>, tensor<?xi32>) -> tensor<?xi32>
  return %out : tensor<?xi32>
}

// CHECK-LABEL: func.func @cx_correlate_valid
// CHECK: ondsp.cx_reduce_mac {{.*}}conjugate
