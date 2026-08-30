// RUN: ondrix-opt %s > %t.ordered.mlir
// RUN: ondrix-opt %t.ordered.mlir --vectorize-ondsp-fixed-elementwise-updates="vector-width=8" > %t.batched.mlir
// RUN: diff %t.ordered.mlir %t.batched.mlir
// RUN: FileCheck %s --implicit-check-not=vector.load --implicit-check-not=vector.store < %t.batched.mlir

// Each function below is the matched update shape with exactly one deviation,
// so the byte-identical diff isolates that deviation as the refusal reason
// (@batch_hand_written_control in elementwise_update_batching.mlir is the
// otherwise identical shape that does batch).

// A dynamic trip count leaves the block count, the covered prefix, and the
// state extent all unknown.
// CHECK-LABEL: func.func @refuse_dynamic_update_count
// CHECK: memref.store
func.func @refuse_dynamic_update_count(%samples: memref<40xi16>, %state: memref<11xi16>,
                                       %base: index, %step: i64, %count: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %tap = %c0 to %count step %c1 {
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

// One extra operation, and a semantic one: the state element is floored before
// the declared update reads it, which is a different update the matcher must
// not silently batch as the declared one.
// CHECK-LABEL: func.func @refuse_extra_body_operation
// CHECK: memref.store
func.func @refuse_extra_body_operation(%samples: memref<40xi16>, %state: memref<11xi16>,
                                       %base: index, %step: i64) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c11 = arith.constant 11 : index
  %floor = arith.constant -16384 : i16
  scf.for %tap = %c0 to %c11 step %c1 {
    %index = arith.subi %base, %tap : index
    %sample = memref.load %samples[%index] : memref<40xi16>
    %wide = arith.extsi %sample : i16 to i64
    %product = arith.muli %step, %wide : i64
    %scaled = ondsp.round_shift %product {scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 15, rounding = nearest_even, overflow = saturate, saturate_to = i16>} : (i64) -> i16
    %element = memref.load %state[%tap] : memref<11xi16>
    %clamped = arith.maxsi %element, %floor : i16
    %element32 = arith.extsi %clamped : i16 to i32
    %scaled32 = arith.extsi %scaled : i16 to i32
    %sum = arith.addi %element32, %scaled32 : i32
    %updated = ondsp.sat_cast %sum {numeric = #ondsp.fixed<signed, storage = i16, frac = 15>} : (i32) -> i16
    memref.store %updated, %state[%tap] : memref<11xi16>
  }
  return
}

// A non-unit stride on the state view: the batched form addresses lanes
// contiguously and this one does not.
// CHECK-LABEL: func.func @refuse_strided_state_view
// CHECK: memref.store
func.func @refuse_strided_state_view(%samples: memref<40xi16>, %storage: memref<22xi16>,
                                     %base: index, %step: i64) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c11 = arith.constant 11 : index
  %state = memref.subview %storage[0] [11] [2] : memref<22xi16> to memref<11xi16, strided<[2]>>
  scf.for %tap = %c0 to %c11 step %c1 {
    %index = arith.subi %base, %tap : index
    %sample = memref.load %samples[%index] : memref<40xi16>
    %wide = arith.extsi %sample : i16 to i64
    %product = arith.muli %step, %wide : i64
    %scaled = ondsp.round_shift %product {scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 15, rounding = nearest_even, overflow = saturate, saturate_to = i16>} : (i64) -> i16
    %element = memref.load %state[%tap] : memref<11xi16, strided<[2]>>
    %element32 = arith.extsi %element : i16 to i32
    %scaled32 = arith.extsi %scaled : i16 to i32
    %sum = arith.addi %element32, %scaled32 : i32
    %updated = ondsp.sat_cast %sum {numeric = #ondsp.fixed<signed, storage = i16, frac = 15>} : (i32) -> i16
    memref.store %updated, %state[%tap] : memref<11xi16, strided<[2]>>
  }
  return
}
