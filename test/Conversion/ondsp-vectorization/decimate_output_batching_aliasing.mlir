// RUN: ondrix-opt %s > %t.ordered.mlir
// RUN: ondrix-opt %t.ordered.mlir --vectorize-ondsp-fixed-decimate-outputs="vector-width=8" > %t.batched.mlir
// RUN: diff %t.ordered.mlir %t.batched.mlir
// RUN: FileCheck %s --implicit-check-not=vector.load --implicit-check-not=vector.store < %t.batched.mlir

// Batching defers a block's W stores past all of that block's K tap loads. The
// ordered schedule makes output m visible to the window of output m + 1; the
// batched one does not. When the output shares storage with an operand the two
// schedules therefore compute different values, so every statically decidable
// sharing is refused.
//
// The shapes below are written directly in memref form rather than through
// `ondrix.fir_decimate`, because the algorithm operation is destination style
// over tensor values and cannot express an output that aliases its input. The
// aliasing enters after bufferization, which is exactly where this pass runs.

// A unit-stride view of the input, offset so that output m lands on the input
// element the window of output m + 5 reads. Resolving the view to its base is
// what catches this.
// CHECK-LABEL: func.func @refuse_output_view_of_input
// CHECK: ondsp.reduce_mac

func.func @refuse_output_view_of_input(%input: memref<48xi16>, %coeffs: memref<8xi16>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c19 = arith.constant 19 : index
  %output = memref.subview %input[10] [19] [1]
      : memref<48xi16> to memref<19xi16, strided<[1], offset: 10>>
  scf.for %index = %c0 to %c19 step %c1 {
    %offset = arith.muli %index, %c2 : index
    %window = memref.subview %input[%offset] [8] [1]
        : memref<48xi16> to memref<8xi16, strided<[1], offset: ?>>
    %zero = ondsp.acc_zero
        : !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
    %reduced = ondsp.reduce_mac %zero, %window, %coeffs {
      numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
      product = #ondsp.product<full>
    } : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>,
         memref<8xi16, strided<[1], offset: ?>>, memref<8xi16>)
        -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
    %sample = ondsp.acc_export %reduced {
      dst = #ondsp.fixed<signed, storage = i16, frac = 15>,
      rounding = #ondsp.rounding<nearest_even>,
      overflow = #ondsp.overflow<saturate>
    } : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>) -> i16
    memref.store %sample, %output[%index] : memref<19xi16, strided<[1], offset: 10>>
  }
  return
}

// The same buffer is both the windowed input and the store target: the output
// value is the input value, with no view in between.
// CHECK-LABEL: func.func @refuse_output_is_input
// CHECK: ondsp.reduce_mac

func.func @refuse_output_is_input(%buffer: memref<44xi16>, %coeffs: memref<8xi16>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c19 = arith.constant 19 : index
  scf.for %index = %c0 to %c19 step %c1 {
    %offset = arith.muli %index, %c2 : index
    %window = memref.subview %buffer[%offset] [8] [1]
        : memref<44xi16> to memref<8xi16, strided<[1], offset: ?>>
    %zero = ondsp.acc_zero
        : !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
    %reduced = ondsp.reduce_mac %zero, %window, %coeffs {
      numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
      product = #ondsp.product<full>
    } : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>,
         memref<8xi16, strided<[1], offset: ?>>, memref<8xi16>)
        -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
    %sample = ondsp.acc_export %reduced {
      dst = #ondsp.fixed<signed, storage = i16, frac = 15>,
      rounding = #ondsp.rounding<nearest_even>,
      overflow = #ondsp.overflow<saturate>
    } : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>) -> i16
    memref.store %sample, %buffer[%index] : memref<44xi16>
  }
  return
}

// The coefficients are a view of the output buffer, so the deferred stores
// would change what a later tap reads even though the input is disjoint.
// CHECK-LABEL: func.func @refuse_coefficients_view_of_output
// CHECK: ondsp.reduce_mac

func.func @refuse_coefficients_view_of_output(%input: memref<44xi16>, %output: memref<19xi16>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c19 = arith.constant 19 : index
  %coeffs = memref.subview %output[0] [8] [1]
      : memref<19xi16> to memref<8xi16, strided<[1]>>
  scf.for %index = %c0 to %c19 step %c1 {
    %offset = arith.muli %index, %c2 : index
    %window = memref.subview %input[%offset] [8] [1]
        : memref<44xi16> to memref<8xi16, strided<[1], offset: ?>>
    %zero = ondsp.acc_zero
        : !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
    %reduced = ondsp.reduce_mac %zero, %window, %coeffs {
      numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
      product = #ondsp.product<full>
    } : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>,
         memref<8xi16, strided<[1], offset: ?>>, memref<8xi16, strided<[1]>>)
        -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
    %sample = ondsp.acc_export %reduced {
      dst = #ondsp.fixed<signed, storage = i16, frac = 15>,
      rounding = #ondsp.rounding<nearest_even>,
      overflow = #ondsp.overflow<saturate>
    } : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>) -> i16
    memref.store %sample, %output[%index] : memref<19xi16>
  }
  return
}

// Two reads of one global are distinct SSA values naming one buffer, so SSA
// identity alone would miss this sharing.
memref.global "private" constant @shared_window : memref<44xi16> = dense<16384>

// CHECK-LABEL: func.func @refuse_same_global_operands
// CHECK: ondsp.reduce_mac

func.func @refuse_same_global_operands(%output: memref<19xi16>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c19 = arith.constant 19 : index
  %source = memref.get_global @shared_window : memref<44xi16>
  %taps = memref.get_global @shared_window : memref<44xi16>
  %coeffs = memref.subview %taps[0] [8] [1]
      : memref<44xi16> to memref<8xi16, strided<[1]>>
  scf.for %index = %c0 to %c19 step %c1 {
    %offset = arith.muli %index, %c2 : index
    %window = memref.subview %source[%offset] [8] [1]
        : memref<44xi16> to memref<8xi16, strided<[1], offset: ?>>
    %zero = ondsp.acc_zero
        : !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
    %reduced = ondsp.reduce_mac %zero, %window, %coeffs {
      numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
      product = #ondsp.product<full>
    } : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>,
         memref<8xi16, strided<[1], offset: ?>>, memref<8xi16, strided<[1]>>)
        -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
    %sample = ondsp.acc_export %reduced {
      dst = #ondsp.fixed<signed, storage = i16, frac = 15>,
      rounding = #ondsp.rounding<nearest_even>,
      overflow = #ondsp.overflow<saturate>
    } : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>) -> i16
    memref.store %sample, %output[%index] : memref<19xi16>
  }
  return
}
