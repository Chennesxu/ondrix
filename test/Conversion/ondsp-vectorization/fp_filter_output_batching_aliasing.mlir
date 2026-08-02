// RUN: ondrix-opt %s > %t.ordered.mlir
// RUN: ondrix-opt %t.ordered.mlir --vectorize-ondsp-fp-filter-outputs="vector-width=8" > %t.batched.mlir
// RUN: diff %t.ordered.mlir %t.batched.mlir
// RUN: FileCheck %s --implicit-check-not=vector.load --implicit-check-not=vector.store < %t.batched.mlir

// Batching defers a block's W stores past all of that block's tap loads. The
// ordered schedule makes output m visible to the window of output m + 1; the
// batched one does not, so every statically decidable storage sharing is
// refused — the same doctrine, analysis, and residual as the fixed-point
// decimate batching. The shapes are written directly in memref form because
// the aliasing enters after bufferization, which is where this pass runs.

// A unit-stride view of the input as the store target.
// CHECK-LABEL: func.func @refuse_output_view_of_input
// CHECK: ondsp.reduce_mac
func.func @refuse_output_view_of_input(%input: memref<40xf32>, %coeffs: memref<8xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c33 = arith.constant 33 : index
  %zero = arith.constant 0.000000e+00 : f32
  %output = memref.subview %input[7] [33] [1]
      : memref<40xf32> to memref<33xf32, strided<[1], offset: 7>>
  scf.for %m = %c0 to %c33 step %c1 {
    %window = memref.subview %input[%m] [8] [1]
        : memref<40xf32> to memref<8xf32, strided<[1], offset: ?>>
    %sample = ondsp.reduce_mac %zero, %window, %coeffs {
      numeric = #ondsp.fp<format = f32, contract = fma>
    } : (f32, memref<8xf32, strided<[1], offset: ?>>, memref<8xf32>) -> f32
    memref.store %sample, %output[%m] : memref<33xf32, strided<[1], offset: 7>>
  }
  return
}

// The same buffer is both the windowed input and the store target.
// CHECK-LABEL: func.func @refuse_output_is_input
// CHECK: ondsp.reduce_mac
func.func @refuse_output_is_input(%buffer: memref<40xf32>, %coeffs: memref<8xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c33 = arith.constant 33 : index
  %zero = arith.constant 0.000000e+00 : f32
  scf.for %m = %c0 to %c33 step %c1 {
    %window = memref.subview %buffer[%m] [8] [1]
        : memref<40xf32> to memref<8xf32, strided<[1], offset: ?>>
    %sample = ondsp.reduce_mac %zero, %window, %coeffs {
      numeric = #ondsp.fp<format = f32, contract = fma>
    } : (f32, memref<8xf32, strided<[1], offset: ?>>, memref<8xf32>) -> f32
    memref.store %sample, %buffer[%m] : memref<40xf32>
  }
  return
}

// Storage chosen at run time is refused rather than assumed distinct.
// CHECK-LABEL: func.func @refuse_selected_output
// CHECK: ondsp.reduce_mac
func.func @refuse_selected_output(%input: memref<40xf32>, %coeffs: memref<8xf32>,
                                  %condition: i1) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c33 = arith.constant 33 : index
  %zero = arith.constant 0.000000e+00 : f32
  %fresh = memref.alloc() : memref<33xf32>
  %aliased = memref.subview %input[7] [33] [1]
      : memref<40xf32> to memref<33xf32, strided<[1], offset: 7>>
  %view = memref.cast %aliased
      : memref<33xf32, strided<[1], offset: 7>> to memref<33xf32, strided<[1], offset: ?>>
  %freshView = memref.cast %fresh : memref<33xf32> to memref<33xf32, strided<[1], offset: ?>>
  %output = arith.select %condition, %view, %freshView : memref<33xf32, strided<[1], offset: ?>>
  scf.for %m = %c0 to %c33 step %c1 {
    %window = memref.subview %input[%m] [8] [1]
        : memref<40xf32> to memref<8xf32, strided<[1], offset: ?>>
    %sample = ondsp.reduce_mac %zero, %window, %coeffs {
      numeric = #ondsp.fp<format = f32, contract = fma>
    } : (f32, memref<8xf32, strided<[1], offset: ?>>, memref<8xf32>) -> f32
    memref.store %sample, %output[%m] : memref<33xf32, strided<[1], offset: ?>>
  }
  return
}

// A non-entry block argument is not the caller-owned residual: the branch
// fixes the aliasing inside the function, so both operands are refused as
// opaque — the provenance rule the shared analysis already enforces.
// CHECK-LABEL: func.func @refuse_branched_operands
// CHECK: ondsp.reduce_mac
func.func @refuse_branched_operands(%buffer: memref<40xf32>, %coeffs: memref<8xf32>) {
  %aliased = memref.subview %buffer[7] [33] [1]
      : memref<40xf32> to memref<33xf32, strided<[1], offset: 7>>
  cf.br ^bb1(%buffer, %aliased : memref<40xf32>, memref<33xf32, strided<[1], offset: 7>>)
^bb1(%input: memref<40xf32>, %output: memref<33xf32, strided<[1], offset: 7>>):
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c33 = arith.constant 33 : index
  %zero = arith.constant 0.000000e+00 : f32
  scf.for %m = %c0 to %c33 step %c1 {
    %window = memref.subview %input[%m] [8] [1]
        : memref<40xf32> to memref<8xf32, strided<[1], offset: ?>>
    %sample = ondsp.reduce_mac %zero, %window, %coeffs {
      numeric = #ondsp.fp<format = f32, contract = fma>
    } : (f32, memref<8xf32, strided<[1], offset: ?>>, memref<8xf32>) -> f32
    memref.store %sample, %output[%m] : memref<33xf32, strided<[1], offset: 7>>
  }
  return
}
