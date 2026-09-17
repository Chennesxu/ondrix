// RUN: ondrix-opt %s --one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map allow-return-allocs" --canonicalize | FileCheck %s

// The adapting state is stored reversed, so the settled samples read ONE
// forward unit-stride window and their tap sum is one reduce_mac.

// CHECK-LABEL: func.func @lms4_q15(
// CHECK-SAME: %[[X:.*]]: memref<16xi16>, %[[D:.*]]: memref<16xi16>, %[[W:.*]]: memref<4xi16>)
// CHECK-NOT: ondrix.lms
// CHECK: %[[ERRORS:.*]] = memref.alloc() {{.*}} : memref<16xi16>
// CHECK: %[[STATE:.*]] = memref.alloc() {{.*}} : memref<4xi16>
// CHECK: scf.for %[[TAP:.*]] = %{{.*}} to %[[C4:.*]] step
// CHECK:   %[[SOURCE:.*]] = arith.subi %[[C3:.*]], %[[TAP]]
// CHECK:   %[[WEIGHT:.*]] = memref.load %[[W]][%[[SOURCE]]]
// CHECK:   memref.store %[[WEIGHT]], %[[STATE]][%[[TAP]]]

// The guarded region evaluates the zero prehistory term by term.
// CHECK: scf.for %{{.*}} = %{{.*}} to %[[C3]] step
// CHECK:   scf.for
// CHECK:     arith.select
// CHECK:     ondsp.mac

// CHECK: scf.for %[[SAMPLE:.*]] = %[[C3]] to %{{.*}} step
// CHECK:   %[[BASE:.*]] = arith.subi %[[SAMPLE]], %[[C3]]
// CHECK:   %[[WINDOW:.*]] = memref.subview %[[X]][%[[BASE]]] [4] [1]
// CHECK:   %[[INITIAL:.*]] = ondsp.acc_zero : <storage = i40, frac = 30, signed, update_overflow = wrap>
// CHECK:   %[[REDUCED:.*]] = ondsp.reduce_mac %[[INITIAL]], %[[WINDOW]], %[[STATE]]
// CHECK:   %[[OUT:.*]] = ondsp.acc_export %[[REDUCED]] {dst = #ondsp.fixed<signed, storage = i16, frac = 15>
// CHECK:   memref.load %[[D]][%[[SAMPLE]]]
// CHECK:   ondsp.sat_cast
// CHECK:   memref.store %{{.*}}, %[[ERRORS]][%[[SAMPLE]]]
// The update walks the same window forward, so no lane of it is reversed.
// CHECK:   scf.for %[[UPDATE:.*]] = %{{.*}} to %[[C4]] step
// CHECK:     memref.load %[[WINDOW]][%[[UPDATE]]]
// CHECK:     memref.store %{{.*}}, %[[STATE]][%[[UPDATE]]]

// CHECK: scf.for %[[MIRROR:.*]] = %{{.*}} to %[[C2:.*]] step
// CHECK:   arith.subi %[[C3]], %[[MIRROR]]
// CHECK: return %[[ERRORS]], %[[STATE]]

func.func @lms4_q15(%x: tensor<16xi16>, %d: tensor<16xi16>, %w: tensor<4xi16>)
    -> (tensor<16xi16>, tensor<4xi16>) {
  %error, %adapted = ondrix.lms %x, %d, %w {
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    rounding = #ondsp.rounding<nearest_even>,
    step_size = 4096 : i64
  } : (tensor<16xi16>, tensor<16xi16>, tensor<4xi16>) -> (tensor<16xi16>, tensor<4xi16>)
  return %error, %adapted : tensor<16xi16>, tensor<4xi16>
}
