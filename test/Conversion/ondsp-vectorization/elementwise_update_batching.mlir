// RUN: ondrix-opt %s --convert-ondrix-to-ondsp --empty-tensor-to-alloc-tensor --one-shot-bufferize="bufferize-function-boundaries allow-return-allocs function-boundary-type-conversion=identity-layout-map" --cse --canonicalize > %t.ordered.mlir
// RUN: ondrix-opt %t.ordered.mlir --vectorize-ondsp-fixed-elementwise-updates="vector-width=8" | FileCheck %s
// RUN: not ondrix-opt %t.ordered.mlir --vectorize-ondsp-fixed-elementwise-updates="vector-width=1" 2>&1 | FileCheck %s --check-prefix=WIDTH
// RUN: not ondrix-opt %t.ordered.mlir --vectorize-ondsp-fixed-elementwise-updates="vector-width=4294967296" 2>&1 | FileCheck %s --check-prefix=WIDE

// Vertical state batching is order preserving — lane k reads and writes only
// state[k] (the pass description carries the full argument).

// WIDTH: vector-width must be greater than one
// WIDE: vector-width must not exceed 4096

// CHECK-LABEL: func.func @lms_q15
// The guarded prehistory update keeps the ordered schedule: its body carries a
// boundary select the matcher does not model.
// CHECK: arith.select
// CHECK: memref.store %{{.*}}, %{{.*}} : memref<11xi16>

// The steady-state update batches taps 0..7 and leaves 8..10 ordered.
// CHECK: %[[LAST:.*]] = arith.constant 7 : index
// CHECK: %[[END:.*]] = arith.constant 8 : index
// CHECK: %[[STEP:.*]] = arith.constant 8 : index
// Both factors are sign extensions of i16, so the exact product fits the
// narrower carrier; the control below keeps i64 on an unextended step.
// CHECK: %[[LANES:.*]] = vector.splat %{{.*}} : vector<8xi32>
// CHECK: scf.for %[[BLOCK:.*]] = %{{.*}} to %[[END]] step %[[STEP]] {
// The sample walk runs backward, so the block loads the span forward from its
// last tap and reverses the lanes onto the ordered tap order.
// CHECK: %[[TOP:.*]] = arith.subi %{{.*}}, %[[BLOCK]] : index
// CHECK: %[[BASE:.*]] = arith.subi %[[TOP]], %[[LAST]] : index
// CHECK: %[[SPAN:.*]] = vector.load %{{.*}}[%[[BASE]]] : memref<40xi16>, vector<8xi16>
// CHECK: vector.shuffle %[[SPAN]], %[[SPAN]] [7, 6, 5, 4, 3, 2, 1, 0] : vector<8xi16>, vector<8xi16>
// CHECK: arith.muli %[[LANES]], %{{.*}} : vector<8xi32>
// CHECK: ondsp.round_shift %{{.*}} : (vector<8xi32>) -> vector<8xi16>
// CHECK: vector.load %{{.*}}[%[[BLOCK]]] : memref<11xi16>, vector<8xi16>
// CHECK: arith.addi %{{.*}} : vector<8xi32>
// CHECK: ondsp.sat_cast %{{.*}} : (vector<8xi32>) -> vector<8xi16>
// CHECK: vector.store %{{.*}}, %{{.*}}[%[[BLOCK]]] : memref<11xi16>, vector<8xi16>
// CHECK: }
// CHECK: scf.for %{{.*}} = %[[END]] to %{{.*}} step %{{.*}} {
// CHECK: ondsp.round_shift %{{.*}} : (i64) -> i16
// CHECK: ondsp.sat_cast %{{.*}} : (i32) -> i16
// CHECK: memref.store %{{.*}}, %{{.*}} : memref<11xi16>

// The hand-written base shape the refusal witnesses deviate from, one
// deviation each; batching it here is what makes those diffs discriminate.
// CHECK-LABEL: func.func @batch_hand_written_control
// CHECK: vector.shuffle %{{.*}} [7, 6, 5, 4, 3, 2, 1, 0] : vector<8xi16>, vector<8xi16>
// Its step arrives as an opaque i64 argument rather than a sign extension, so
// no width bounds the product and the declared carrier stays.
// CHECK: arith.muli %{{.*}} : vector<8xi64>
// CHECK: vector.store %{{.*}} : memref<11xi16>, vector<8xi16>

// One bit past what the narrowed carrier holds: i16 x i17 needs 33 bits, so
// this batches with the declared carrier while @lms_q15's i16 x i16 narrows.
// CHECK-LABEL: func.func @batch_step_one_bit_too_wide
// CHECK: vector.shuffle %{{.*}} [7, 6, 5, 4, 3, 2, 1, 0] : vector<8xi16>, vector<8xi16>
// CHECK: arith.muli %{{.*}} : vector<8xi64>
// CHECK: vector.store %{{.*}} : memref<11xi16>, vector<8xi16>

func.func @lms_q15(%x: tensor<40xi16>, %d: tensor<40xi16>, %w: tensor<11xi16>)
    -> (tensor<40xi16>, tensor<11xi16>) {
  %error, %adapted = ondrix.lms %x, %d, %w {
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    rounding = #ondsp.rounding<nearest_even>,
    step_size = 4096 : i64
  } : (tensor<40xi16>, tensor<40xi16>, tensor<11xi16>) -> (tensor<40xi16>, tensor<11xi16>)
  return %error, %adapted : tensor<40xi16>, tensor<11xi16>
}

func.func @batch_hand_written_control(%samples: memref<40xi16>, %state: memref<11xi16>,
                                      %base: index, %step: i64) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c11 = arith.constant 11 : index
  scf.for %tap = %c0 to %c11 step %c1 {
    %index = arith.subi %base, %tap : index
    %sample = memref.load %samples[%index] : memref<40xi16>
    %wide = arith.extsi %sample : i16 to i64
    %product = arith.muli %step, %wide : i64
    %scaled = ondsp.round_shift %product {scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 15, rounding = nearest_even, overflow = saturate, saturate_to = i16>} : (i64) -> i16
    %element = memref.load %state[%tap] : memref<11xi16>
    %element32 = arith.extsi %element : i16 to i32
    %scaled32 = arith.extsi %scaled : i16 to i32
    %sum = arith.addi %element32, %scaled32 : i32
    %updated = ondsp.sat_cast %sum {numeric = #ondsp.fixed<signed, storage = i16, frac = 15>} : (i32) -> i16
    memref.store %updated, %state[%tap] : memref<11xi16>
  }
  return
}

func.func @batch_step_one_bit_too_wide(%samples: memref<40xi16>, %state: memref<11xi16>,
                                       %base: index, %narrow: i17) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c11 = arith.constant 11 : index
  %step = arith.extsi %narrow : i17 to i64
  scf.for %tap = %c0 to %c11 step %c1 {
    %index = arith.subi %base, %tap : index
    %sample = memref.load %samples[%index] : memref<40xi16>
    %wide = arith.extsi %sample : i16 to i64
    %product = arith.muli %step, %wide : i64
    %scaled = ondsp.round_shift %product {scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 15, rounding = nearest_even, overflow = saturate, saturate_to = i16>} : (i64) -> i16
    %element = memref.load %state[%tap] : memref<11xi16>
    %element32 = arith.extsi %element : i16 to i32
    %scaled32 = arith.extsi %scaled : i16 to i32
    %sum = arith.addi %element32, %scaled32 : i32
    %updated = ondsp.sat_cast %sum {numeric = #ondsp.fixed<signed, storage = i16, frac = 15>} : (i32) -> i16
    memref.store %updated, %state[%tap] : memref<11xi16>
  }
  return
}
