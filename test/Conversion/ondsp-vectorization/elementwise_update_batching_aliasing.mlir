// RUN: ondrix-opt %s > %t.ordered.mlir
// RUN: ondrix-opt %t.ordered.mlir --vectorize-ondsp-fixed-elementwise-updates="vector-width=8" > %t.batched.mlir
// RUN: diff %t.ordered.mlir %t.batched.mlir
// RUN: FileCheck %s --implicit-check-not=vector.load --implicit-check-not=vector.store < %t.batched.mlir

// Batching widens a block's sample loads and defers that block's stores past
// them. The ordered schedule makes the store into state[k] visible to the
// sample load of index k + 1; the batched one does not. Every statically
// decidable sharing is therefore refused, and so is every base the analysis
// cannot place (@batch_hand_written_control in elementwise_update_batching.mlir
// is the same shape with a placeable, distinct state buffer).

// The one buffer serving as both sequences.
// CHECK-LABEL: func.func @refuse_state_is_samples
// CHECK: memref.store
func.func @refuse_state_is_samples(%shared: memref<40xi16>, %base: index, %step: i64) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c11 = arith.constant 11 : index
  scf.for %tap = %c0 to %c11 step %c1 {
    %index = arith.subi %base, %tap : index
    %sample = memref.load %shared[%index] : memref<40xi16>
    %wide = arith.extsi %sample : i16 to i64
    %product = arith.muli %step, %wide : i64
    %scaled = ondsp.round_shift %product {scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 15, rounding = nearest_even, overflow = saturate, saturate_to = i16>} : (i64) -> i16
    %element = memref.load %shared[%tap] : memref<40xi16>
    %element32 = arith.extsi %element : i16 to i32
    %scaled32 = arith.extsi %scaled : i16 to i32
    %sum = arith.addi %element32, %scaled32 : i32
    %updated = ondsp.sat_cast %sum {numeric = #ondsp.fixed<signed, storage = i16, frac = 15>} : (i32) -> i16
    memref.store %updated, %shared[%tap] : memref<40xi16>
  }
  return
}

// A unit-stride view of the sample buffer, offset so that the state elements
// land where later indices read samples. Resolving the view to its base is
// what catches this.
// CHECK-LABEL: func.func @refuse_state_view_of_samples
// CHECK: memref.store
func.func @refuse_state_view_of_samples(%samples: memref<40xi16>, %base: index, %step: i64) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c11 = arith.constant 11 : index
  %state = memref.subview %samples[4] [11] [1]
      : memref<40xi16> to memref<11xi16, strided<[1], offset: 4>>
  scf.for %tap = %c0 to %c11 step %c1 {
    %index = arith.subi %base, %tap : index
    %sample = memref.load %samples[%index] : memref<40xi16>
    %wide = arith.extsi %sample : i16 to i64
    %product = arith.muli %step, %wide : i64
    %scaled = ondsp.round_shift %product {scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 15, rounding = nearest_even, overflow = saturate, saturate_to = i16>} : (i64) -> i16
    %element = memref.load %state[%tap] : memref<11xi16, strided<[1], offset: 4>>
    %element32 = arith.extsi %element : i16 to i32
    %scaled32 = arith.extsi %scaled : i16 to i32
    %sum = arith.addi %element32, %scaled32 : i32
    %updated = ondsp.sat_cast %sum {numeric = #ondsp.fixed<signed, storage = i16, frac = 15>} : (i32) -> i16
    memref.store %updated, %state[%tap] : memref<11xi16, strided<[1], offset: 4>>
  }
  return
}

// A state buffer chosen at run time. The select is not a view, so nothing
// places its result, and an opaque producer may hand back storage the sample
// argument already names.
// CHECK-LABEL: func.func @refuse_opaque_state_producer
// CHECK: memref.store
func.func @refuse_opaque_state_producer(%samples: memref<40xi16>, %left: memref<11xi16>,
                                        %right: memref<11xi16>, %pick: i1, %base: index,
                                        %step: i64) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c11 = arith.constant 11 : index
  %state = arith.select %pick, %left, %right : memref<11xi16>
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
