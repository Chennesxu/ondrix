// RUN: ondrix-opt %s --tile-ondrix-fir-filter="tile-size=4" | FileCheck %s --check-prefix=TILE4
// RUN: ondrix-opt %s --tile-ondrix-fir-filter="tile-size=1" | FileCheck %s --check-prefix=TILE1

// TILE4-LABEL: func.func @dynamic_q15
// TILE4: %[[STEP:.*]] = arith.constant 4 : index
// TILE4-COUNT-3: cf.assert
// TILE4: %[[OUTPUT_SIZE:.*]] = tensor.dim %[[INIT:.*]], %{{.*}}
// TILE4: %[[RESULT:.*]] = scf.for %[[OFFSET:.*]] = %{{.*}} to %[[OUTPUT_SIZE]] step %[[STEP]] iter_args(%[[DEST:.*]] = %[[INIT]])
// TILE4-NOT: cf.assert
// TILE4: %[[TILE_SIZE:.*]] = affine.min
// TILE4: %[[COEFF_SIZE:.*]] = tensor.dim %[[COEFFS:.*]], %{{.*}}
// TILE4: %[[HALO:.*]] = arith.subi %[[COEFF_SIZE]], %{{.*}}
// TILE4: %[[HALO_SIZE:.*]] = arith.addi %[[TILE_SIZE]], %[[HALO]]
// TILE4: %[[INPUT_TILE:.*]] = tensor.extract_slice %[[INPUT:.*]][%[[OFFSET]]] [%[[HALO_SIZE]]] [1]
// TILE4: %[[INIT_TILE:.*]] = tensor.extract_slice %[[DEST]][%[[OFFSET]]] [%[[TILE_SIZE]]] [1]
// TILE4: %[[TILED:.*]] = ondrix.fir_filter %[[INPUT_TILE]], %[[COEFFS]], %[[INIT_TILE]]
// TILE4-SAME: accumulator = !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
// TILE4-SAME: boundary = #ondrix.fir_boundary<valid>
// TILE4-SAME: product = #ondsp.product<full>
// TILE4: tensor.insert_slice %[[TILED]] into %[[DEST]][%[[OFFSET]]] [%[[TILE_SIZE]]] [1]
// TILE4-NOT: cf.assert
// TILE4: return %[[RESULT]]
func.func @dynamic_q15(
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

// TILE1-LABEL: func.func @static_f32
// TILE1: %[[STEP:.*]] = arith.constant 1 : index
// TILE1: scf.for %[[OFFSET:.*]] = %{{.*}} to %{{.*}} step %[[STEP]]
// TILE1: tensor.extract_slice %{{.*}}[%[[OFFSET]]] [3] [1]
// TILE1: tensor.extract_slice %{{.*}}[%[[OFFSET]]] [1] [1]
// TILE1: ondrix.fir_filter
// TILE1-SAME: numeric = #ondsp.fp<format = f32, contract = fma>
// TILE1: tensor.insert_slice
func.func @static_f32(
    %input: tensor<8xf32>, %coeffs: tensor<3xf32>, %init: tensor<6xf32>)
    -> tensor<6xf32> {
  %result = ondrix.fir_filter %input, %coeffs, %init {
    boundary = #ondrix.fir_boundary<valid>,
    numeric = #ondsp.fp<format = f32, contract = fma>
  } : (tensor<8xf32>, tensor<3xf32>, tensor<6xf32>) -> tensor<6xf32>
  return %result : tensor<6xf32>
}
