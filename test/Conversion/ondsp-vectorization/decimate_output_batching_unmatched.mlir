// RUN: ondrix-opt %s --one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map" --cse --canonicalize > %t.ordered.mlir
// RUN: ondrix-opt %t.ordered.mlir --vectorize-ondsp-fixed-decimate-outputs="vector-width=8" > %t.batched.mlir
// The pass must fail closed, not fail soft: a loop it does not fully recognize
// comes out byte-identical, so an unmatched program keeps exactly the ordered
// schedule rather than a partially batched one.
// RUN: diff %t.ordered.mlir %t.batched.mlir
// RUN: FileCheck %s --implicit-check-not=vector.load --implicit-check-not=vector.store < %t.batched.mlir

// Every refusal condition that a whole function can express, each with the
// ordered reduction pinned as the surviving schedule.

// A factor of three does not make the span covering W outputs a single
// contiguous run with a fixed even-lane shuffle, so slice one refuses it.
// CHECK-LABEL: func.func @refuse_factor_three
// CHECK: ondsp.reduce_mac

func.func @refuse_factor_three(
    %input: tensor<62xi16>, %coeffs: tensor<8xi16>, %init: tensor<19xi16>) -> tensor<19xi16> {
  %result = ondrix.fir_decimate %input, %coeffs, %init {
    accumulator = !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>,
    dst = #ondsp.fixed<signed, storage = i16, frac = 15>,
    factor = 3,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    overflow = #ondsp.overflow<saturate>,
    product = #ondsp.product<full>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<62xi16>, tensor<8xi16>, tensor<19xi16>) -> tensor<19xi16>
  return %result : tensor<19xi16>
}

// Dynamic extents leave both the batched trip count and the in-bounds span
// unknown, so the loop stays ordered.
// CHECK-LABEL: func.func @refuse_dynamic_extents
// CHECK: ondsp.reduce_mac

func.func @refuse_dynamic_extents(
    %input: tensor<?xi16>, %coeffs: tensor<?xi16>, %init: tensor<?xi16>) -> tensor<?xi16> {
  %result = ondrix.fir_decimate %input, %coeffs, %init {
    accumulator = !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>,
    dst = #ondsp.fixed<signed, storage = i16, frac = 15>,
    factor = 2,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    overflow = #ondsp.overflow<saturate>,
    product = #ondsp.product<full>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<?xi16>, tensor<?xi16>, tensor<?xi16>) -> tensor<?xi16>
  return %result : tensor<?xi16>
}

// A valid-boundary FIR bufferizes to a loop with the same accumulator, the same
// reduction, the same export and the same store, differing only in that the
// window offset is the output index itself rather than a scaled one. The
// matcher must not touch it: batching it would need a different span and a
// different lane extraction.
// CHECK-LABEL: func.func @refuse_unit_stride_fir_filter
// CHECK: ondsp.reduce_mac

func.func @refuse_unit_stride_fir_filter(
    %input: tensor<24xi16>, %coeffs: tensor<8xi16>, %init: tensor<17xi16>) -> tensor<17xi16> {
  %result = ondrix.fir_filter %input, %coeffs, %init {
    accumulator = !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>,
    boundary = #ondrix.fir_boundary<valid>,
    dst = #ondsp.fixed<signed, storage = i16, frac = 15>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    overflow = #ondsp.overflow<saturate>,
    product = #ondsp.product<full>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<24xi16>, tensor<8xi16>, tensor<17xi16>) -> tensor<17xi16>
  return %result : tensor<17xi16>
}

// Exactly one block of outputs leaves nothing to batch: the block that holds
// the final output always stays on the ordered loop, because its contiguous
// factor * width span at the last tap would end one element past what the
// ordered schedule reads.
// CHECK-LABEL: func.func @refuse_single_short_block
// CHECK: ondsp.reduce_mac

func.func @refuse_single_short_block(
    %input: tensor<22xi16>, %coeffs: tensor<8xi16>, %init: tensor<8xi16>) -> tensor<8xi16> {
  %result = ondrix.fir_decimate %input, %coeffs, %init {
    accumulator = !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>,
    dst = #ondsp.fixed<signed, storage = i16, frac = 15>,
    factor = 2,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    overflow = #ondsp.overflow<saturate>,
    product = #ondsp.product<full>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<22xi16>, tensor<8xi16>, tensor<8xi16>) -> tensor<8xi16>
  return %result : tensor<8xi16>
}

// A non-default memory space is outside what the Vector to LLVM lowering
// accepts, so the batched load and store are not available even though the
// arithmetic and the extents are.
// CHECK-LABEL: func.func @refuse_nonzero_memory_space
// CHECK: ondsp.reduce_mac

func.func @refuse_nonzero_memory_space(
    %input: memref<44xi16, 3>, %coeffs: memref<8xi16, 3>, %output: memref<19xi16, 3>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c19 = arith.constant 19 : index
  scf.for %index = %c0 to %c19 step %c1 {
    %offset = arith.muli %index, %c2 : index
    %window = memref.subview %input[%offset] [8] [1]
        : memref<44xi16, 3> to memref<8xi16, strided<[1], offset: ?>, 3>
    %zero = ondsp.acc_zero
        : !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
    %reduced = ondsp.reduce_mac %zero, %window, %coeffs {
      numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
      product = #ondsp.product<full>
    } : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>,
         memref<8xi16, strided<[1], offset: ?>, 3>, memref<8xi16, 3>)
        -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
    %sample = ondsp.acc_export %reduced {
      dst = #ondsp.fixed<signed, storage = i16, frac = 15>,
      rounding = #ondsp.rounding<nearest_even>,
      overflow = #ondsp.overflow<saturate>
    } : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>) -> i16
    memref.store %sample, %output[%index] : memref<19xi16, 3>
  }
  return
}

// A non-unit window stride is not the phase-zero window the contract declares,
// and the batched span assumes contiguity.
// CHECK-LABEL: func.func @refuse_strided_window
// CHECK: ondsp.reduce_mac

func.func @refuse_strided_window(
    %input: memref<88xi16>, %coeffs: memref<8xi16>, %output: memref<19xi16>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c19 = arith.constant 19 : index
  scf.for %index = %c0 to %c19 step %c1 {
    %offset = arith.muli %index, %c2 : index
    %window = memref.subview %input[%offset] [8] [2]
        : memref<88xi16> to memref<8xi16, strided<[2], offset: ?>>
    %zero = ondsp.acc_zero
        : !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
    %reduced = ondsp.reduce_mac %zero, %window, %coeffs {
      numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
      product = #ondsp.product<full>
    } : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>,
         memref<8xi16, strided<[2], offset: ?>>, memref<8xi16>)
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

// The exported sample has a second consumer, so the ordered chain is not the
// single-use chain the rewrite replaces and the loop stays as it is. The
// function also carries an already batched accumulator beside the ordered loop,
// which the matcher must neither widen again nor confuse with the loop it is
// looking at.
// CHECK-LABEL: func.func @refuse_extra_sample_store
// CHECK: ondsp.reduce_mac

func.func @refuse_extra_sample_store(
    %input: memref<44xi16>, %coeffs: memref<8xi16>, %output: memref<19xi16>,
    %value: vector<8xi16>, %coefficient: i16) -> vector<8xi16> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c19 = arith.constant 19 : index
  %batchedZero = ondsp.acc_zero
      : !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate, lanes = 8>
  %batched = ondsp.mac %batchedZero, %value, %coefficient {
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    product = #ondsp.product<full>
  } : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate, lanes = 8>,
       vector<8xi16>, i16)
      -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate, lanes = 8>
  %batchedResult = ondsp.acc_export %batched {
    dst = #ondsp.fixed<signed, storage = i16, frac = 15>,
    rounding = #ondsp.rounding<nearest_even>,
    overflow = #ondsp.overflow<saturate>
  } : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate, lanes = 8>)
      -> vector<8xi16>
  // The loop below carries a second, unrelated store of the exported sample, so
  // the export chain is not single use and the matcher refuses the loop.
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
         memref<8xi16, strided<[1], offset: ?>>, memref<8xi16>)
        -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
    %sample = ondsp.acc_export %reduced {
      dst = #ondsp.fixed<signed, storage = i16, frac = 15>,
      rounding = #ondsp.rounding<nearest_even>,
      overflow = #ondsp.overflow<saturate>
    } : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>) -> i16
    memref.store %sample, %output[%index] : memref<19xi16>
    memref.store %sample, %output[%c0] : memref<19xi16>
  }
  return %batchedResult : vector<8xi16>
}
