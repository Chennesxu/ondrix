// RUN: ondrix-opt %s --one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map" | FileCheck %s

// CHECK-LABEL: func.func @q15_full_tile(
// CHECK-SAME: %[[INPUT:.*]]: memref<?xi16>, %[[COEFFS:.*]]: memref<?xi16>, %[[OUTPUT:.*]]: memref<?xi16>, %[[ORIGIN:.*]]: index)
// CHECK: cf.assert {{.*}}, "full FIR requires at least one input sample"
// CHECK: cf.assert {{.*}}, "full FIR requires at least one coefficient"
// CHECK: arith.cmpi ule, %[[ORIGIN]]
// CHECK: cf.assert {{.*}}, "full FIR output tile must lie within the complete output range"
// CHECK-NOT: ondrix.fir_filter
func.func @q15_full_tile(
    %input: tensor<?xi16>, %coeffs: tensor<?xi16>, %init: tensor<?xi16>,
    %origin: index) -> tensor<?xi16> {
  %result = ondrix.fir_filter %input, %coeffs, %init, %origin {
    accumulator = !ondsp.acc<storage = i40, frac = 30, signed,
                              update_overflow = saturate>,
    boundary = #ondrix.fir_boundary<full>,
    dst = #ondsp.fixed<signed, storage = i16, frac = 15>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    overflow = #ondsp.overflow<saturate>,
    product = #ondsp.product<full>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<?xi16>, tensor<?xi16>, tensor<?xi16>, index) -> tensor<?xi16>
  return %result : tensor<?xi16>
}
