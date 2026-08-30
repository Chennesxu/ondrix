// RUN: ondrix-opt %s --vectorize-ondsp-fixed-window-mac-reduce="vector-width=8" > %t.mlir
// RUN: FileCheck %s --input-file=%t.mlir

// Each function below deviates from @batches_base in exactly one respect, and
// each deviation alone is what makes the rewrite fail closed.

// CHECK-LABEL: func.func @batches_base
// CHECK: vector.shuffle
// CHECK: vector.reduction <add>
func.func @batches_base(%samples: memref<40xi16>, %coeffs: memref<8xi16>, %base: index) -> i64 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %seed = arith.constant 0 : i64
  %sum = scf.for %k = %c0 to %c8 step %c1 iter_args(%acc = %seed) -> (i64) {
    %i = arith.subi %base, %k : index
    %s = memref.load %samples[%i] : memref<40xi16>
    %c = memref.load %coeffs[%k] : memref<8xi16>
    %sw = arith.extsi %s : i16 to i64
    %cw = arith.extsi %c : i16 to i64
    %p = arith.muli %cw, %sw : i64
    %n = arith.addi %acc, %p : i64
    scf.yield %n : i64
  }
  return %sum : i64
}

// A FORWARD sample walk is a different reduction, not this one: the span load
// plus lane reversal would then place every term on the wrong lane.
// CHECK-LABEL: func.func @forward_sample_walk_refused
// CHECK-NOT: vector.shuffle
// CHECK-NOT: vector.reduction
func.func @forward_sample_walk_refused(%samples: memref<40xi16>, %coeffs: memref<8xi16>,
                                       %base: index) -> i64 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %seed = arith.constant 0 : i64
  %sum = scf.for %k = %c0 to %c8 step %c1 iter_args(%acc = %seed) -> (i64) {
    %i = arith.addi %base, %k : index
    %s = memref.load %samples[%i] : memref<40xi16>
    %c = memref.load %coeffs[%k] : memref<8xi16>
    %sw = arith.extsi %s : i16 to i64
    %cw = arith.extsi %c : i16 to i64
    %p = arith.muli %cw, %sw : i64
    %n = arith.addi %acc, %p : i64
    scf.yield %n : i64
  }
  return %sum : i64
}

// i32 x i48 needs 80 bits, which no accepted carrier has, so the product
// would not be exact on every term and the reduction keeps its schedule.
// CHECK-LABEL: func.func @product_exceeds_every_carrier_refused
// CHECK-NOT: vector.shuffle
// CHECK-NOT: vector.reduction
func.func @product_exceeds_every_carrier_refused(%samples: memref<40xi32>,
                                                 %coeffs: memref<8xi48>, %base: index) -> i64 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %seed = arith.constant 0 : i64
  %sum = scf.for %k = %c0 to %c8 step %c1 iter_args(%acc = %seed) -> (i64) {
    %i = arith.subi %base, %k : index
    %s = memref.load %samples[%i] : memref<40xi32>
    %c = memref.load %coeffs[%k] : memref<8xi48>
    %sw = arith.extsi %s : i32 to i64
    %cw = arith.extsi %c : i48 to i64
    %p = arith.muli %cw, %sw : i64
    %n = arith.addi %acc, %p : i64
    scf.yield %n : i64
  }
  return %sum : i64
}

// One extra operation in the body defeats the exact cover, so the matcher can
// no longer claim it understands every effect the loop has.
// CHECK-LABEL: func.func @extra_body_operation_refused
// CHECK-NOT: vector.shuffle
// CHECK-NOT: vector.reduction
func.func @extra_body_operation_refused(%samples: memref<40xi16>, %coeffs: memref<8xi16>,
                                        %scratch: memref<8xi16>, %base: index) -> i64 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %seed = arith.constant 0 : i64
  %sum = scf.for %k = %c0 to %c8 step %c1 iter_args(%acc = %seed) -> (i64) {
    %i = arith.subi %base, %k : index
    %s = memref.load %samples[%i] : memref<40xi16>
    %c = memref.load %coeffs[%k] : memref<8xi16>
    memref.store %c, %scratch[%k] : memref<8xi16>
    %sw = arith.extsi %s : i16 to i64
    %cw = arith.extsi %c : i16 to i64
    %p = arith.muli %cw, %sw : i64
    %n = arith.addi %acc, %p : i64
    scf.yield %n : i64
  }
  return %sum : i64
}

// A dynamic extent leaves no static span the block could be shown to stay
// inside, so the walk base cannot be placed.
// CHECK-LABEL: func.func @dynamic_sample_extent_refused
// CHECK-NOT: vector.shuffle
// CHECK-NOT: vector.reduction
func.func @dynamic_sample_extent_refused(%samples: memref<?xi16>, %coeffs: memref<8xi16>,
                                         %base: index) -> i64 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %seed = arith.constant 0 : i64
  %sum = scf.for %k = %c0 to %c8 step %c1 iter_args(%acc = %seed) -> (i64) {
    %i = arith.subi %base, %k : index
    %s = memref.load %samples[%i] : memref<?xi16>
    %c = memref.load %coeffs[%k] : memref<8xi16>
    %sw = arith.extsi %s : i16 to i64
    %cw = arith.extsi %c : i16 to i64
    %p = arith.muli %cw, %sw : i64
    %n = arith.addi %acc, %p : i64
    scf.yield %n : i64
  }
  return %sum : i64
}

// The candidate carrier is clamped to the accumulator: i4 x i4 fits the
// narrow carrier but extending i32 down to i8 would not even be valid IR.
// CHECK-LABEL: func.func @batch_narrow_accumulator
// CHECK: arith.muli %{{.*}} : vector<8xi8>
// CHECK-NOT: arith.extsi %{{.*}} : vector<8xi32> to vector<8xi8>
// CHECK: vector.reduction <add>, %{{.*}} : vector<8xi8> into i8
func.func @batch_narrow_accumulator(%samples: memref<40xi4>, %coeffs: memref<8xi4>,
                                    %base: index) -> i8 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %seed = arith.constant 0 : i8
  %sum = scf.for %k = %c0 to %c8 step %c1 iter_args(%acc = %seed) -> (i8) {
    %i = arith.subi %base, %k : index
    %s = memref.load %samples[%i] : memref<40xi4>
    %c = memref.load %coeffs[%k] : memref<8xi4>
    %sw = arith.extsi %s : i4 to i8
    %cw = arith.extsi %c : i4 to i8
    %p = arith.muli %cw, %sw : i8
    %n = arith.addi %acc, %p : i8
    scf.yield %n : i8
  }
  return %sum : i8
}
