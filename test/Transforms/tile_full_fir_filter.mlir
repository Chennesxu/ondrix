// RUN: ondrix-opt %s --tile-ondrix-fir-filter="tile-size=4" | FileCheck %s

// CHECK-LABEL: func.func @dynamic_q15_full
// CHECK: %[[STEP:.*]] = arith.constant 4 : index
// CHECK-COUNT-3: cf.assert
// CHECK: %[[OUTPUT_SIZE:.*]] = tensor.dim %[[INIT:.*]], %{{.*}}
// CHECK: %[[RESULT:.*]] = scf.for %[[ORIGIN:.*]] = %{{.*}} to %[[OUTPUT_SIZE]] step %[[STEP]] iter_args(%[[DEST:.*]] = %[[INIT]])
// CHECK-NOT: cf.assert
// CHECK: %[[TILE_SIZE:.*]] = affine.min
// CHECK: %[[INIT_TILE:.*]] = tensor.extract_slice %[[DEST]][%[[ORIGIN]]] [%[[TILE_SIZE]]] [1]
// CHECK: %[[TILED:.*]] = ondrix.fir_filter %[[INPUT:.*]], %[[COEFFS:.*]], %[[INIT_TILE]], %[[ORIGIN]]
// CHECK-SAME: accumulator = !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
// CHECK-SAME: boundary = #ondrix.fir_boundary<full>
// CHECK-SAME: product = #ondsp.product<full>
// CHECK: tensor.insert_slice %[[TILED]] into %[[DEST]][%[[ORIGIN]]] [%[[TILE_SIZE]]] [1]
// CHECK-NOT: cf.assert
// CHECK: return %[[RESULT]]
func.func @dynamic_q15_full(
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
