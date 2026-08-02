// RUN: ondrix-opt %s --convert-ondrix-to-ondsp --vectorize-ondsp-fixed-memref-reduce="vector-width=4" --normalize-ondsp-fixed-vector-reduce --lower-ondsp-f32-reduce-to-scalar > %t.lowered.mlir
// RUN: FileCheck %s --check-prefix=LOWERED < %t.lowered.mlir
// RUN: ondrix-opt %t.lowered.mlir --convert-ondsp-fixed-to-scalar --expand-strided-metadata --lower-affine --convert-scf-to-cf --convert-vector-to-llvm --finalize-memref-to-llvm --convert-math-to-llvm --convert-arith-to-llvm --convert-cf-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts > %t.mlir
// RUN: ondrix-translate %t.mlir --mlir-to-llvmir > %t.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.ll -o %t.o
// RUN: cc -ffp-contract=off %S/Inputs/q31_f32_full_output_prototype_aot.c %t.o -lm -o %t
// RUN: %t

// LOWERED-LABEL: func.func @q31_full_output_vector
// LOWERED: scf.for
// LOWERED: memref.subview
// LOWERED: scf.for
// LOWERED: vector.load
// LOWERED: ondsp.acc_add_term
// LOWERED-NOT: ondsp.reduce_mac
// LOWERED-LABEL: func.func @q31_full_output_raw_vector
// LOWERED: vector.load
// LOWERED: ondsp.acc_add_term
// LOWERED-NOT: ondsp.reduce_mac
// LOWERED-LABEL: func.func @f32_full_output_scalar
// LOWERED-NOT: vector.load
// LOWERED: scf.for
// LOWERED-NOT: vector.load
// LOWERED: memref.subview
// LOWERED-NOT: vector.load
// LOWERED: scf.for
// LOWERED-NOT: vector.load
// LOWERED: memref.load
// LOWERED-NOT: vector.load
// LOWERED: math.fma
// LOWERED-NOT: vector.load
// LOWERED-NOT: ondsp.reduce_mac
// LOWERED-LABEL: func.func @f32_full_output_scalar_off
// LOWERED-NOT: vector.load
// LOWERED: arith.mulf
// LOWERED: arith.addf
// LOWERED-NOT: math.fma
// LOWERED-NOT: ondsp.reduce_mac

func.func @q31_full_output_vector(
    %input: memref<?xi32, strided<[1], offset: ?>>,
    %coefficients: memref<?xi32, strided<[1], offset: ?>>,
    %output: memref<?xi32, strided<[1], offset: ?>>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %input_length = memref.dim %input, %c0
      : memref<?xi32, strided<[1], offset: ?>>
  %coefficient_length = memref.dim %coefficients, %c0
      : memref<?xi32, strided<[1], offset: ?>>
  %output_length = memref.dim %output, %c0
      : memref<?xi32, strided<[1], offset: ?>>
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
        : memref<?xi32, strided<[1], offset: ?>>
          to memref<?xi32, strided<[1], offset: ?>>
    %accumulator = ondrix.fir %window, %coefficients {
      numeric = #ondsp.fixed<signed, storage = i32, frac = 31>,
      product = #ondsp.product<full>
    } : (memref<?xi32, strided<[1], offset: ?>>,
         memref<?xi32, strided<[1], offset: ?>>)
        -> !ondsp.acc<storage = i64, frac = 62, signed,
                      update_overflow = saturate>
    %result = ondsp.acc_export %accumulator {
      dst = #ondsp.fixed<signed, storage = i32, frac = 31>,
      rounding = #ondsp.rounding<nearest_even>,
      overflow = #ondsp.overflow<saturate>
    } : (!ondsp.acc<storage = i64, frac = 62, signed,
                         update_overflow = saturate>) -> i32
    memref.store %result, %output[%output_index]
        : memref<?xi32, strided<[1], offset: ?>>
  }
  return
}

// This test-only raw result observes accumulator policy differences that a
// saturating Q31 export can hide.
func.func @q31_full_output_raw_vector(
    %input: memref<?xi32, strided<[1], offset: ?>>,
    %coefficients: memref<?xi32, strided<[1], offset: ?>>,
    %output: memref<?xi64, strided<[1], offset: ?>>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %input_length = memref.dim %input, %c0
      : memref<?xi32, strided<[1], offset: ?>>
  %coefficient_length = memref.dim %coefficients, %c0
      : memref<?xi32, strided<[1], offset: ?>>
  %output_length = memref.dim %output, %c0
      : memref<?xi64, strided<[1], offset: ?>>
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
        : memref<?xi32, strided<[1], offset: ?>>
          to memref<?xi32, strided<[1], offset: ?>>
    %accumulator = ondrix.fir %window, %coefficients {
      numeric = #ondsp.fixed<signed, storage = i32, frac = 31>,
      product = #ondsp.product<full>
    } : (memref<?xi32, strided<[1], offset: ?>>,
         memref<?xi32, strided<[1], offset: ?>>)
        -> !ondsp.acc<storage = i64, frac = 62, signed,
                      update_overflow = saturate>
    %raw = builtin.unrealized_conversion_cast %accumulator
        : !ondsp.acc<storage = i64, frac = 62, signed,
                     update_overflow = saturate> to i64
    memref.store %raw, %output[%output_index]
        : memref<?xi64, strided<[1], offset: ?>>
  }
  return
}

func.func @f32_full_output_scalar(
    %input: memref<?xf32, strided<[?], offset: ?>>,
    %coefficients: memref<?xf32, strided<[?], offset: ?>>,
    %output: memref<?xf32, strided<[?], offset: ?>>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %input_length = memref.dim %input, %c0
      : memref<?xf32, strided<[?], offset: ?>>
  %coefficient_length = memref.dim %coefficients, %c0
      : memref<?xf32, strided<[?], offset: ?>>
  %output_length = memref.dim %output, %c0
      : memref<?xf32, strided<[?], offset: ?>>
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
        : memref<?xf32, strided<[?], offset: ?>>
          to memref<?xf32, strided<[?], offset: ?>>
    %result = ondrix.fir %window, %coefficients {
      numeric = #ondsp.fp<format = f32, contract = fma>
    } : (memref<?xf32, strided<[?], offset: ?>>,
         memref<?xf32, strided<[?], offset: ?>>) -> f32
    memref.store %result, %output[%output_index]
        : memref<?xf32, strided<[?], offset: ?>>
  }
  return
}

// The off contract rounds every tap product before the accumulator observes
// it, so this twin must not be lowered through a fused update. It pairs with
// the fused function above to gate both f32 contract modes over the same
// strided windows.
func.func @f32_full_output_scalar_off(
    %input: memref<?xf32, strided<[?], offset: ?>>,
    %coefficients: memref<?xf32, strided<[?], offset: ?>>,
    %output: memref<?xf32, strided<[?], offset: ?>>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %input_length = memref.dim %input, %c0
      : memref<?xf32, strided<[?], offset: ?>>
  %coefficient_length = memref.dim %coefficients, %c0
      : memref<?xf32, strided<[?], offset: ?>>
  %output_length = memref.dim %output, %c0
      : memref<?xf32, strided<[?], offset: ?>>
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
        : memref<?xf32, strided<[?], offset: ?>>
          to memref<?xf32, strided<[?], offset: ?>>
    %result = ondrix.fir %window, %coefficients {
      numeric = #ondsp.fp<format = f32, contract = off>
    } : (memref<?xf32, strided<[?], offset: ?>>,
         memref<?xf32, strided<[?], offset: ?>>) -> f32
    memref.store %result, %output[%output_index]
        : memref<?xf32, strided<[?], offset: ?>>
  }
  return
}
