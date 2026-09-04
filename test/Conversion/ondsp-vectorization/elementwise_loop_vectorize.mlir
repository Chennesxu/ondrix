// RUN: ondrix-opt %s --vectorize-ondsp-fixed-elementwise-loops="vector-width=8" | FileCheck %s

// The batched block is the ordered iterations themselves: element i reads only
// in[i] and writes only out[i], so the lanes fold nothing.

// CHECK-LABEL: func.func @batch_requantize
// CHECK: %[[STEP:.*]] = arith.constant 8 : index
// CHECK: scf.for %[[BLOCK:.*]] = %{{.*}} to %{{.*}} step %[[STEP]]
// CHECK: %[[LANES:.*]] = vector.load %arg0[%[[BLOCK]]] : memref<4096xi32>, vector<8xi32>
// CHECK: %[[SCALED:.*]] = ondsp.round_shift %[[LANES]] {scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 6, rounding = nearest_even, overflow = saturate, saturate_to = i32>} : (vector<8xi32>) -> vector<8xi32>
// CHECK: vector.store %[[SCALED]], %arg1[%[[BLOCK]]] : memref<4096xi32>, vector<8xi32>
// CHECK-NOT: memref.load
func.func @batch_requantize(%in: memref<4096xi32>, %out: memref<4096xi32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4096 = arith.constant 4096 : index
  scf.for %i = %c0 to %c4096 step %c1 {
    %element = memref.load %in[%i] : memref<4096xi32>
    %scaled = ondsp.round_shift %element {scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 6, rounding = nearest_even, overflow = saturate, saturate_to = i32>} : (i32) -> i32
    memref.store %scaled, %out[%i] : memref<4096xi32>
  }
  return
}

// A width-changing chain travels lane by lane, and the five leftover elements
// stay on the untouched ordered loop.
// CHECK-LABEL: func.func @batch_narrowing_chain_with_remainder
// CHECK: %[[BATCHED:.*]] = arith.constant 24 : index
// CHECK: scf.for %{{.*}} = %{{.*}} to %[[BATCHED]] step %{{.*}}
// CHECK: vector.load {{.*}} : memref<29xi32>, vector<8xi32>
// CHECK: ondsp.round_shift {{.*}} : (vector<8xi32>) -> vector<8xi16>
// CHECK: arith.extsi {{.*}} : vector<8xi16> to vector<8xi32>
// CHECK: vector.store {{.*}} : memref<29xi32>, vector<8xi32>
// CHECK: scf.for %{{.*}} = %[[BATCHED]] to %{{.*}} step %{{.*}}
// CHECK: memref.load
// CHECK: memref.store
func.func @batch_narrowing_chain_with_remainder(%in: memref<29xi32>, %out: memref<29xi32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c29 = arith.constant 29 : index
  scf.for %i = %c0 to %c29 step %c1 {
    %element = memref.load %in[%i] : memref<29xi32>
    %scaled = ondsp.round_shift %element {scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 16, rounding = nearest_even, overflow = saturate, saturate_to = i16>} : (i32) -> i16
    %wide = arith.extsi %scaled : i16 to i32
    memref.store %wide, %out[%i] : memref<29xi32>
  }
  return
}

// The block's store is deferred past its loads, so one buffer read and written
// under the same name would be read at a different time than the ordered
// schedule reads it.
// CHECK-LABEL: func.func @refuse_aliasing_sequences
// CHECK-NOT: vector.load
// CHECK: memref.store
func.func @refuse_aliasing_sequences(%buffer: memref<64xi32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c64 = arith.constant 64 : index
  scf.for %i = %c0 to %c64 step %c1 {
    %element = memref.load %buffer[%i] : memref<64xi32>
    %scaled = ondsp.round_shift %element {scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 6, rounding = nearest_even, overflow = saturate, saturate_to = i32>} : (i32) -> i32
    memref.store %scaled, %buffer[%i] : memref<64xi32>
  }
  return
}

// A second operand makes the body something other than an elementwise map of
// one sequence, and the matcher must not guess how its lanes pair.
// CHECK-LABEL: func.func @refuse_non_elementwise_body
// CHECK-NOT: vector.load
// CHECK: memref.store
func.func @refuse_non_elementwise_body(%in: memref<64xi32>, %out: memref<64xi32>, %bias: i32) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c64 = arith.constant 64 : index
  scf.for %i = %c0 to %c64 step %c1 {
    %element = memref.load %in[%i] : memref<64xi32>
    %biased = arith.addi %element, %bias : i32
    %scaled = ondsp.round_shift %biased {scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 6, rounding = nearest_even, overflow = saturate, saturate_to = i32>} : (i32) -> i32
    memref.store %scaled, %out[%i] : memref<64xi32>
  }
  return
}

// A non-unit stride makes a contiguous lane load a different element set.
// CHECK-LABEL: func.func @refuse_strided_sequence
// CHECK-NOT: vector.load
// CHECK: memref.store
func.func @refuse_strided_sequence(%in: memref<64xi32, strided<[2], offset: 0>>,
                                   %out: memref<64xi32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c64 = arith.constant 64 : index
  scf.for %i = %c0 to %c64 step %c1 {
    %element = memref.load %in[%i] : memref<64xi32, strided<[2], offset: 0>>
    %scaled = ondsp.round_shift %element {scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 6, rounding = nearest_even, overflow = saturate, saturate_to = i32>} : (i32) -> i32
    memref.store %scaled, %out[%i] : memref<64xi32>
  }
  return
}
