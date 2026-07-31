// RUN: ondrix-opt %s --one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map" --cse --canonicalize --vectorize-ondsp-fixed-decimate-outputs="vector-width=8" | FileCheck %s

// Regression pins for where the batched loop stops.
//
// Two independent parities decide the boundary and are pinned in all four
// combinations: whether `N - K` is even (so the last window ends exactly at the
// last input element) or odd (so one input element is never read), and whether
// the output length is a multiple of the batch width.
//
// The batched loop covers `((M - 1) / W) * W` outputs, NOT `(M / W) * W`. The
// block that would hold the final output always stays on the ordered loop,
// because its contiguous `factor * W` span at the last tap would end one
// element past what the ordered schedule reads. When `W` does not divide `M`
// the two formulas agree and the rule costs nothing; when it does, a whole
// block of outputs stays ordered, and that is the case worth pinning because
// it is the one an "obvious" `M / W` would get wrong.
//
// Every function below also pins that the ordered remainder starts exactly at
// the batched loop's upper bound, so no output is computed twice or skipped.

// N - K = 30 is even and M = 16 is a multiple of the width, so the batched loop
// stops at 8 and the LAST EIGHT outputs stay on the ordered loop.
// CHECK-LABEL: func.func @even_extent_multiple_outputs
// CHECK-SAME: memref<38xi16>, %{{.*}}: memref<8xi16>, %{{.*}}: memref<16xi16>
// CHECK: %[[TOTAL:.*]] = arith.constant 16 : index
// CHECK: %[[BATCHED_END:.*]] = arith.constant 8 : index
// CHECK: %[[BATCH_STEP:.*]] = arith.constant 8 : index
// CHECK: scf.for %{{.*}} = %{{.*}} to %[[BATCHED_END]] step %[[BATCH_STEP]] {
// CHECK: vector.store {{.*}} : memref<16xi16>, vector<8xi16>
// CHECK: scf.for %{{.*}} = %[[BATCHED_END]] to %[[TOTAL]] step %{{.*}} {
// CHECK: ondsp.reduce_mac
// CHECK: memref.store {{.*}} : memref<16xi16>

func.func @even_extent_multiple_outputs(
    %input: tensor<38xi16>, %coeffs: tensor<8xi16>, %init: tensor<16xi16>) -> tensor<16xi16> {
  %result = ondrix.fir_decimate %input, %coeffs, %init {
    accumulator = !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>,
    dst = #ondsp.fixed<signed, storage = i16, frac = 15>,
    factor = 2,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    overflow = #ondsp.overflow<saturate>,
    product = #ondsp.product<full>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<38xi16>, tensor<8xi16>, tensor<16xi16>) -> tensor<16xi16>
  return %result : tensor<16xi16>
}

// N - K = 36 is even and M = 19 is not a multiple of the width, so both block
// formulas agree at 16 and three outputs remain ordered.
// CHECK-LABEL: func.func @even_extent_remainder_outputs
// CHECK: %[[TOTAL:.*]] = arith.constant 19 : index
// CHECK: %[[BATCHED_END:.*]] = arith.constant 16 : index
// CHECK: %[[BATCH_STEP:.*]] = arith.constant 8 : index
// CHECK: scf.for %{{.*}} = %{{.*}} to %[[BATCHED_END]] step %[[BATCH_STEP]] {
// CHECK: vector.store {{.*}} : memref<19xi16>, vector<8xi16>
// CHECK: scf.for %{{.*}} = %[[BATCHED_END]] to %[[TOTAL]] step %{{.*}} {
// CHECK: ondsp.reduce_mac

func.func @even_extent_remainder_outputs(
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

// N - K = 31 is odd, so the final input element is never read by any window;
// M = 16 is still a multiple of the width, so the last eight outputs still stay
// ordered. The extra slack must not change where the batched loop stops.
// CHECK-LABEL: func.func @odd_extent_multiple_outputs
// CHECK-SAME: memref<39xi16>
// CHECK: %[[TOTAL:.*]] = arith.constant 16 : index
// CHECK: %[[BATCHED_END:.*]] = arith.constant 8 : index
// CHECK: %[[BATCH_STEP:.*]] = arith.constant 8 : index
// CHECK: scf.for %{{.*}} = %{{.*}} to %[[BATCHED_END]] step %[[BATCH_STEP]] {
// CHECK: vector.store {{.*}} : memref<16xi16>, vector<8xi16>
// CHECK: scf.for %{{.*}} = %[[BATCHED_END]] to %[[TOTAL]] step %{{.*}} {
// CHECK: ondsp.reduce_mac

func.func @odd_extent_multiple_outputs(
    %input: tensor<39xi16>, %coeffs: tensor<8xi16>, %init: tensor<16xi16>) -> tensor<16xi16> {
  %result = ondrix.fir_decimate %input, %coeffs, %init {
    accumulator = !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>,
    dst = #ondsp.fixed<signed, storage = i16, frac = 15>,
    factor = 2,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    overflow = #ondsp.overflow<saturate>,
    product = #ondsp.product<full>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<39xi16>, tensor<8xi16>, tensor<16xi16>) -> tensor<16xi16>
  return %result : tensor<16xi16>
}

// N - K = 37 is odd and M = 19 is not a multiple of the width.
// CHECK-LABEL: func.func @odd_extent_remainder_outputs
// CHECK-SAME: memref<45xi16>
// CHECK: %[[TOTAL:.*]] = arith.constant 19 : index
// CHECK: %[[BATCHED_END:.*]] = arith.constant 16 : index
// CHECK: %[[BATCH_STEP:.*]] = arith.constant 8 : index
// CHECK: scf.for %{{.*}} = %{{.*}} to %[[BATCHED_END]] step %[[BATCH_STEP]] {
// CHECK: vector.store {{.*}} : memref<19xi16>, vector<8xi16>
// CHECK: scf.for %{{.*}} = %[[BATCHED_END]] to %[[TOTAL]] step %{{.*}} {
// CHECK: ondsp.reduce_mac

func.func @odd_extent_remainder_outputs(
    %input: tensor<45xi16>, %coeffs: tensor<8xi16>, %init: tensor<19xi16>) -> tensor<19xi16> {
  %result = ondrix.fir_decimate %input, %coeffs, %init {
    accumulator = !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>,
    dst = #ondsp.fixed<signed, storage = i16, frac = 15>,
    factor = 2,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    overflow = #ondsp.overflow<saturate>,
    product = #ondsp.product<full>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<45xi16>, tensor<8xi16>, tensor<19xi16>) -> tensor<19xi16>
  return %result : tensor<19xi16>
}
