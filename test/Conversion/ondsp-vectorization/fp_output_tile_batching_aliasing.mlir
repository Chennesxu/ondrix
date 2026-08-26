// RUN: ondrix-opt %s > %t.ordered.mlir
// RUN: ondrix-opt %t.ordered.mlir --vectorize-ondsp-fp-filter-outputs="vector-width=4 supports-vector-fma=true" > %t.batched.mlir
// RUN: diff %t.ordered.mlir %t.batched.mlir

// The tile emitter hoists every value read above a block's W stores, so any
// storage sharing between the store target and either read sequence is
// refused; the diff proves the whole file is left untouched.

// The same buffer is the value sequence and the store target: the ordered
// program feeds output k back into later outputs' x[0].
func.func @refuse_in_place(%buf: memref<8xf32>, %table: memref<8x8xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  scf.for %k = %c0 to %c8 step %c1 {
    %x0 = memref.load %buf[%c0] : memref<8xf32>
    %w0 = memref.load %table[%c0, %k] : memref<8x8xf32>
    %seed = arith.mulf %x0, %w0 : f32
    %sum = scf.for %n = %c1 to %c8 step %c1 iter_args(%acc = %seed) -> (f32) {
      %x = memref.load %buf[%n] : memref<8xf32>
      %w = memref.load %table[%n, %k] : memref<8x8xf32>
      %p = arith.mulf %x, %w : f32
      %a = arith.addf %acc, %p : f32
      scf.yield %a : f32
    }
    memref.store %sum, %buf[%k] : memref<8xf32>
  }
  return
}

// The store target is a unit-stride view of the value buffer.
func.func @refuse_output_view_of_values(%buf: memref<16xf32>, %table: memref<8x8xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %output = memref.subview %buf[8] [8] [1] : memref<16xf32> to memref<8xf32, strided<[1], offset: 8>>
  scf.for %k = %c0 to %c8 step %c1 {
    %x0 = memref.load %buf[%c0] : memref<16xf32>
    %w0 = memref.load %table[%c0, %k] : memref<8x8xf32>
    %seed = arith.mulf %x0, %w0 : f32
    %sum = scf.for %n = %c1 to %c8 step %c1 iter_args(%acc = %seed) -> (f32) {
      %x = memref.load %buf[%n] : memref<16xf32>
      %w = memref.load %table[%n, %k] : memref<8x8xf32>
      %p = arith.mulf %x, %w : f32
      %a = arith.addf %acc, %p : f32
      scf.yield %a : f32
    }
    memref.store %sum, %output[%k] : memref<8xf32, strided<[1], offset: 8>>
  }
  return
}

// The store target shares storage with the coefficient table.
func.func @refuse_output_view_of_table(%x: memref<8xf32>, %table: memref<9x8xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %terms = memref.subview %table[0, 0] [8, 8] [1, 1] : memref<9x8xf32> to memref<8x8xf32, strided<[8, 1]>>
  %output = memref.subview %table[8, 0] [1, 8] [1, 1] : memref<9x8xf32> to memref<8xf32, strided<[1], offset: 64>>
  scf.for %k = %c0 to %c8 step %c1 {
    %x0 = memref.load %x[%c0] : memref<8xf32>
    %w0 = memref.load %terms[%c0, %k] : memref<8x8xf32, strided<[8, 1]>>
    %seed = arith.mulf %x0, %w0 : f32
    %sum = scf.for %n = %c1 to %c8 step %c1 iter_args(%acc = %seed) -> (f32) {
      %xv = memref.load %x[%n] : memref<8xf32>
      %w = memref.load %terms[%n, %k] : memref<8x8xf32, strided<[8, 1]>>
      %p = arith.mulf %xv, %w : f32
      %a = arith.addf %acc, %p : f32
      scf.yield %a : f32
    }
    memref.store %sum, %output[%k] : memref<8xf32, strided<[1], offset: 64>>
  }
  return
}
