// RUN: ondrix-opt %s --one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map" --cse --canonicalize > %t.ordered.mlir
// RUN: ondrix-opt %t.ordered.mlir --vectorize-ondsp-fixed-decimate-outputs="vector-width=8" | FileCheck %s
// RUN: not ondrix-opt %t.ordered.mlir --vectorize-ondsp-fixed-decimate-outputs="vector-width=1" 2>&1 | FileCheck %s --check-prefix=WIDTH

// Vertical output batching: the lanes of the accumulator carry independent
// outputs, each folding the same taps in the same increasing order into its own
// accumulator. Nothing is reassociated and no lane is ever combined with
// another, so the rewrite needs no range or overflow proof — the contrast with
// the horizontal reduction passes, whose lanes do reorder the fold, is the
// point of keeping the two axes separate.
//
// With nineteen outputs and width eight the batched loop covers outputs 0..15
// and the untouched ordered loop covers 16..18.

// WIDTH: vector-width must be greater than one

// CHECK-LABEL: func.func @batch_static_factor_two
// The batched loop steps by the vector width over the full blocks.
// CHECK: %[[BATCHED_END:.*]] = arith.constant 16 : index
// CHECK: %[[BATCH_STEP:.*]] = arith.constant 8 : index
// CHECK: scf.for %[[BLOCK:.*]] = %{{.*}} to %[[BATCHED_END]] step %[[BATCH_STEP]] {
// CHECK: ondsp.acc_zero : <storage = i40, frac = 30, signed, update_overflow = saturate, lanes = 8>
// CHECK: %[[WINDOW:.*]] = arith.muli %[[BLOCK]], %{{.*}} : index
// Tap zero reads the contiguous factor * width span covering all eight outputs
// and keeps the phase-zero elements.
// CHECK: %[[SPAN:.*]] = vector.load %{{.*}}[%[[WINDOW]]] : memref<44xi16>, vector<16xi16>
// CHECK: vector.shuffle %[[SPAN]], %[[SPAN]] [0, 2, 4, 6, 8, 10, 12, 14] : vector<16xi16>, vector<16xi16>
// The coefficient stays scalar; the per-lane broadcast is declared semantics.
// CHECK: memref.load
// CHECK: ondsp.mac
// CHECK-SAME: update_overflow = saturate, lanes = 8>, vector<8xi16>, i16)
// CHECK-SAME: update_overflow = saturate, lanes = 8>
// One span load and one shuffle per tap: seven more after tap zero.
// CHECK-COUNT-7: vector.load {{.*}} : memref<44xi16>, vector<16xi16>
// CHECK: ondsp.acc_export
// CHECK-SAME: update_overflow = saturate, lanes = 8>) -> vector<8xi16>
// CHECK: vector.store {{.*}} : memref<19xi16>, vector<8xi16>
// The remaining three outputs keep the ordered reduction, unchanged, starting
// exactly where the batched loop stopped.
// CHECK: scf.for %{{.*}} = %[[BATCHED_END]] to %{{.*}} step %{{.*}} {
// CHECK: memref.subview
// CHECK: ondsp.acc_zero : <storage = i40, frac = 30, signed, update_overflow = saturate>
// CHECK: ondsp.reduce_mac
// CHECK: ondsp.acc_export
// CHECK-SAME: update_overflow = saturate>) -> i16
// CHECK: memref.store

func.func @batch_static_factor_two(
    %input: tensor<44xi16>, %coeffs: tensor<8xi16>, %init: tensor<19xi16>) -> tensor<19xi16> {
  %result = ondrix.fir_decimate %input, %coeffs, %init {
    accumulator = !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>,
    dst = #ondsp.fixed<signed, storage = i16, frac = 15>,
    factor = 2,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    overflow = #ondsp.overflow<saturate>,
    product = #ondsp.product<full>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<44xi16>, tensor<8xi16>, tensor<19xi16>) -> tensor<19xi16>
  return %result : tensor<19xi16>
}

// A wrapping accumulator takes the same rewrite: order preservation, not the
// overflow policy, is what makes the batching legal. Nothing here consults a
// reassociation class.

// CHECK-LABEL: func.func @batch_wrapping_accumulator
// CHECK: ondsp.mac
// CHECK-SAME: !ondsp.acc<storage = i34, frac = 30, signed, update_overflow = wrap, lanes = 8>
// CHECK: vector.store {{.*}} : memref<11xi16>, vector<8xi16>

func.func @batch_wrapping_accumulator(
    %input: tensor<28xi16>, %coeffs: tensor<8xi16>, %init: tensor<11xi16>) -> tensor<11xi16> {
  %result = ondrix.fir_decimate %input, %coeffs, %init {
    accumulator = !ondsp.acc<storage = i34, frac = 30, signed, update_overflow = wrap>,
    dst = #ondsp.fixed<signed, storage = i16, frac = 15>,
    factor = 2,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    overflow = #ondsp.overflow<saturate>,
    product = #ondsp.product<full>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<28xi16>, tensor<8xi16>, tensor<11xi16>) -> tensor<11xi16>
  return %result : tensor<11xi16>
}
