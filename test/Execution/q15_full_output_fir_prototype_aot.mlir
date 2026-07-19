// RUN: ondrix-opt %s --convert-ondrix-to-ondsp --vectorize-ondsp-fixed-memref-reduce="vector-width=4" --normalize-ondsp-fixed-vector-reduce > %t.vector.mlir
// RUN: FileCheck %s --check-prefix=VECTOR < %t.vector.mlir
// RUN: ondrix-opt %t.vector.mlir --convert-ondsp-fixed-to-scalar --expand-strided-metadata --lower-affine --convert-scf-to-cf --convert-vector-to-llvm --finalize-memref-to-llvm --convert-arith-to-llvm --convert-cf-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts > %t.mlir
// RUN: ondrix-translate %t.mlir --mlir-to-llvmir > %t.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.ll -o %t.o
// RUN: cc %S/Inputs/q15_full_output_fir_prototype_aot.c %t.o -o %t
// RUN: %t
// RUN: cc %S/Inputs/q15_full_output_fir_prototype_mismatch.c %t.o -o %t.mismatch
// RUN: not --crash %t.mismatch empty
// RUN: not --crash %t.mismatch short
// RUN: not --crash %t.mismatch output
// RUN: not --crash %t.mismatch full
// RUN: not --crash %t.mismatch input
// RUN: not --crash %t.mismatch coefficients

// This executable prototype keeps full-output orchestration in upstream
// dialects. A future destination-style Ondrix operation must make the current
// no-alias calling precondition explicit and verifiable.

// VECTOR-LABEL: func.func @q15_full_output_vector
// VECTOR: scf.for
// VECTOR: memref.subview
// VECTOR: scf.for
// VECTOR: vector.load
// VECTOR: ondsp.acc_add_term
// VECTOR-NOT: ondsp.reduce_mac
// VECTOR-LABEL: func.func @q15_full_output_scalar
// VECTOR: scf.for
// VECTOR: memref.subview
// VECTOR: ondsp.reduce_mac
// VECTOR-NOT: vector.load
// VECTOR-LABEL: func.func @q15_full_boundary_scalar
// VECTOR: scf.for
// VECTOR: scf.for
// VECTOR: scf.if
// VECTOR: ondsp.mac
// VECTOR-NOT: ondsp.reduce_mac

func.func @q15_full_output_vector(
    %input: memref<?xi16, strided<[1], offset: ?>>,
    %coefficients: memref<?xi16, strided<[1], offset: ?>>,
    %output: memref<?xi16, strided<[1], offset: ?>>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %input_length = memref.dim %input, %c0
      : memref<?xi16, strided<[1], offset: ?>>
  %coefficient_length = memref.dim %coefficients, %c0
      : memref<?xi16, strided<[1], offset: ?>>
  %output_length = memref.dim %output, %c0
      : memref<?xi16, strided<[1], offset: ?>>
  %has_coefficients = arith.cmpi ugt, %coefficient_length, %c0 : index
  cf.assert %has_coefficients, "valid FIR requires at least one coefficient"
  %input_covers_window = arith.cmpi uge, %input_length, %coefficient_length : index
  cf.assert %input_covers_window, "valid FIR input must cover one coefficient window"
  %remaining = arith.subi %input_length, %coefficient_length : index
  %required_output_length = arith.addi %remaining, %c1 : index
  %output_matches = arith.cmpi eq, %output_length, %required_output_length : index
  cf.assert %output_matches, "valid FIR output length must equal input length minus coefficient length plus one"

  scf.for %output_index = %c0 to %output_length step %c1 {
    %window = memref.subview %input[%output_index] [%coefficient_length] [1]
        : memref<?xi16, strided<[1], offset: ?>>
          to memref<?xi16, strided<[1], offset: ?>>
    %accumulator = ondrix.fir %window, %coefficients {
      numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
      product = #ondsp.product<full>
    } : (memref<?xi16, strided<[1], offset: ?>>,
         memref<?xi16, strided<[1], offset: ?>>)
        -> !ondsp.acc<storage = i40, frac = 30, signed,
                      update_overflow = saturate>
    %result = ondsp.acc_export %accumulator {
      dst = #ondsp.fixed<signed, storage = i16, frac = 15>,
      rounding = #ondsp.rounding<toward_negative>,
      overflow = #ondsp.overflow<wrap>
    } : (!ondsp.acc<storage = i40, frac = 30, signed,
                         update_overflow = saturate>) -> i16
    memref.store %result, %output[%output_index]
        : memref<?xi16, strided<[1], offset: ?>>
  }
  return
}

func.func @q15_full_output_scalar(
    %input: memref<?xi16, strided<[?], offset: ?>>,
    %coefficients: memref<?xi16, strided<[?], offset: ?>>,
    %output: memref<?xi16, strided<[?], offset: ?>>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %input_length = memref.dim %input, %c0
      : memref<?xi16, strided<[?], offset: ?>>
  %coefficient_length = memref.dim %coefficients, %c0
      : memref<?xi16, strided<[?], offset: ?>>
  %output_length = memref.dim %output, %c0
      : memref<?xi16, strided<[?], offset: ?>>
  %has_coefficients = arith.cmpi ugt, %coefficient_length, %c0 : index
  cf.assert %has_coefficients, "valid FIR requires at least one coefficient"
  %input_covers_window = arith.cmpi uge, %input_length, %coefficient_length : index
  cf.assert %input_covers_window, "valid FIR input must cover one coefficient window"
  %remaining = arith.subi %input_length, %coefficient_length : index
  %required_output_length = arith.addi %remaining, %c1 : index
  %output_matches = arith.cmpi eq, %output_length, %required_output_length : index
  cf.assert %output_matches, "valid FIR output length must equal input length minus coefficient length plus one"

  scf.for %output_index = %c0 to %output_length step %c1 {
    %window = memref.subview %input[%output_index] [%coefficient_length] [1]
        : memref<?xi16, strided<[?], offset: ?>>
          to memref<?xi16, strided<[?], offset: ?>>
    %accumulator = ondrix.fir %window, %coefficients {
      numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
      product = #ondsp.product<full>
    } : (memref<?xi16, strided<[?], offset: ?>>,
         memref<?xi16, strided<[?], offset: ?>>)
        -> !ondsp.acc<storage = i40, frac = 30, signed,
                      update_overflow = saturate>
    %result = ondsp.acc_export %accumulator {
      dst = #ondsp.fixed<signed, storage = i16, frac = 15>,
      rounding = #ondsp.rounding<toward_negative>,
      overflow = #ondsp.overflow<wrap>
    } : (!ondsp.acc<storage = i40, frac = 30, signed,
                         update_overflow = saturate>) -> i16
    memref.store %result, %output[%output_index]
        : memref<?xi16, strided<[?], offset: ?>>
  }
  return
}

func.func @q15_full_boundary_scalar(
    %input: memref<?xi16, strided<[?], offset: ?>>,
    %coefficients: memref<?xi16, strided<[?], offset: ?>>,
    %output: memref<?xi16, strided<[?], offset: ?>>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %input_length = memref.dim %input, %c0
      : memref<?xi16, strided<[?], offset: ?>>
  %coefficient_length = memref.dim %coefficients, %c0
      : memref<?xi16, strided<[?], offset: ?>>
  %output_length = memref.dim %output, %c0
      : memref<?xi16, strided<[?], offset: ?>>
  %has_input = arith.cmpi ugt, %input_length, %c0 : index
  cf.assert %has_input, "full FIR requires at least one input sample"
  %has_coefficients = arith.cmpi ugt, %coefficient_length, %c0 : index
  cf.assert %has_coefficients, "full FIR requires at least one coefficient"
  %combined_length = arith.addi %input_length, %coefficient_length : index
  %required_output_length = arith.subi %combined_length, %c1 : index
  %output_matches = arith.cmpi eq, %output_length, %required_output_length : index
  cf.assert %output_matches, "full FIR output length must equal input length plus coefficient length minus one"
  %left_padding = arith.subi %coefficient_length, %c1 : index

  scf.for %output_index = %c0 to %output_length step %c1 {
    %zero = ondsp.acc_zero
        : !ondsp.acc<storage = i40, frac = 30, signed,
                      update_overflow = saturate>
    %accumulator = scf.for %tap = %c0 to %coefficient_length step %c1
        iter_args(%current = %zero)
        -> (!ondsp.acc<storage = i40, frac = 30, signed,
                      update_overflow = saturate>) {
      %padded_index = arith.addi %output_index, %tap : index
      %past_left_padding = arith.cmpi uge, %padded_index, %left_padding : index
      %input_index = arith.subi %padded_index, %left_padding : index
      %before_right_padding = arith.cmpi ult, %input_index, %input_length : index
      %in_bounds = arith.andi %past_left_padding, %before_right_padding : i1
      %next = scf.if %in_bounds
          -> (!ondsp.acc<storage = i40, frac = 30, signed,
                         update_overflow = saturate>) {
        %input_value = memref.load %input[%input_index]
            : memref<?xi16, strided<[?], offset: ?>>
        %coefficient = memref.load %coefficients[%tap]
            : memref<?xi16, strided<[?], offset: ?>>
        %updated = ondsp.mac %current, %input_value, %coefficient {
          numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
          product = #ondsp.product<full>
        } : (!ondsp.acc<storage = i40, frac = 30, signed,
                           update_overflow = saturate>, i16, i16)
            -> !ondsp.acc<storage = i40, frac = 30, signed,
                          update_overflow = saturate>
        scf.yield %updated
            : !ondsp.acc<storage = i40, frac = 30, signed,
                          update_overflow = saturate>
      } else {
        scf.yield %current
            : !ondsp.acc<storage = i40, frac = 30, signed,
                          update_overflow = saturate>
      }
      scf.yield %next
          : !ondsp.acc<storage = i40, frac = 30, signed,
                        update_overflow = saturate>
    }
    %result = ondsp.acc_export %accumulator {
      dst = #ondsp.fixed<signed, storage = i16, frac = 15>,
      rounding = #ondsp.rounding<toward_negative>,
      overflow = #ondsp.overflow<wrap>
    } : (!ondsp.acc<storage = i40, frac = 30, signed,
                         update_overflow = saturate>) -> i16
    memref.store %result, %output[%output_index]
        : memref<?xi16, strided<[?], offset: ?>>
  }
  return
}
