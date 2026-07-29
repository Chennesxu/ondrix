// RUN: ondrix-opt %s --convert-ondrix-to-ondsp --vectorize-ondsp-fixed-memref-reduce="vector-width=8" --parallelize-ondsp-fixed-wrap-vector-reduce --normalize-ondsp-fixed-vector-reduce | FileCheck %s --check-prefix=PLAN
// RUN: ondrix-opt %s --convert-ondrix-to-ondsp --vectorize-ondsp-fixed-memref-reduce="vector-width=8" --parallelize-ondsp-fixed-wrap-vector-reduce --normalize-ondsp-fixed-vector-reduce --convert-ondsp-fixed-to-scalar --convert-scf-to-cf --convert-vector-to-llvm --finalize-memref-to-llvm --convert-arith-to-llvm --convert-cf-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts > %t.mlir
// RUN: ondrix-translate %t.mlir --mlir-to-llvmir > %t.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.ll -o %t.o
// RUN: cc %S/Inputs/q15_i48_wrap_reduce_aot.c %t.o -o %t
// RUN: %t

// A genuinely wrapping i48 accumulator.
//
// The committed wrap-path evidence so far only covers accumulators that a
// range proof shows can never wrap, so the exact-modulo argument was never
// executed against real wraparound. Reassociation legality does NOT depend
// on whether a wrap occurs: modular addition is associative and commutative
// at every width, so the horizontal Vector schedule and the ordered scalar
// schedule are equal mod 2^48 either way. This gate is the committed
// executable evidence for the case where the wrap actually happens (the
// repository precedent is a reviewer verifying i48 by hand; this makes it a
// regression gate).
//
// All four kernels declare the same accumulator contract and differ only in
// the memref layout and the export destination. The identity-layout operands
// admit the wrap horizontal path; the dynamic-stride operands fall back to
// the ordered per-element fold. The Q15 pair pins equality at the declared
// destination boundary, and the raw pair exports the identity signed
// i64/frac30 reading of the accumulator so the gate compares the RAW folded
// i48 value rather than only its saturating Q15 projection.

// PLAN-LABEL: func.func @q15_i48_wrap_horizontal
// PLAN: vector.reduction <add>
// PLAN: ondsp.acc_add_term
// PLAN-SAME: !ondsp.acc<storage = i48, frac = 30, signed, update_overflow = wrap>

// PLAN-LABEL: func.func @q15_i48_wrap_ordered
// PLAN: ondsp.reduce_mac
// PLAN-NOT: vector.reduction

// PLAN-LABEL: func.func @q15_i48_wrap_raw_horizontal
// PLAN: vector.reduction <add>
// PLAN: ondsp.acc_add_term
// PLAN-SAME: !ondsp.acc<storage = i48, frac = 30, signed, update_overflow = wrap>
// PLAN: ondsp.acc_export
// PLAN-SAME: dst = #ondsp.fixed<signed, storage = i64, frac = 30>

// PLAN-LABEL: func.func @q15_i48_wrap_raw_ordered
// PLAN: ondsp.reduce_mac
// PLAN-NOT: vector.reduction

func.func @q15_i48_wrap_horizontal(
    %lhs: memref<?xi16>, %rhs: memref<?xi16>) -> i16 {
  %zero = ondsp.acc_zero
      : !ondsp.acc<storage = i48, frac = 30, signed, update_overflow = wrap>
  %acc = ondsp.reduce_mac %zero, %lhs, %rhs {
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    product = #ondsp.product<full>
  } : (!ondsp.acc<storage = i48, frac = 30, signed, update_overflow = wrap>, memref<?xi16>, memref<?xi16>) -> !ondsp.acc<storage = i48, frac = 30, signed, update_overflow = wrap>
  %result = ondsp.acc_export %acc {
    dst = #ondsp.fixed<signed, storage = i16, frac = 15>,
    rounding = #ondsp.rounding<nearest_even>,
    overflow = #ondsp.overflow<saturate>
  } : (!ondsp.acc<storage = i48, frac = 30, signed, update_overflow = wrap>) -> i16
  return %result : i16
}

func.func @q15_i48_wrap_ordered(
    %lhs: memref<?xi16, strided<[?], offset: ?>>,
    %rhs: memref<?xi16, strided<[?], offset: ?>>) -> i16 {
  %zero = ondsp.acc_zero
      : !ondsp.acc<storage = i48, frac = 30, signed, update_overflow = wrap>
  %acc = ondsp.reduce_mac %zero, %lhs, %rhs {
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    product = #ondsp.product<full>
  } : (!ondsp.acc<storage = i48, frac = 30, signed, update_overflow = wrap>, memref<?xi16, strided<[?], offset: ?>>, memref<?xi16, strided<[?], offset: ?>>) -> !ondsp.acc<storage = i48, frac = 30, signed, update_overflow = wrap>
  %result = ondsp.acc_export %acc {
    dst = #ondsp.fixed<signed, storage = i16, frac = 15>,
    rounding = #ondsp.rounding<nearest_even>,
    overflow = #ondsp.overflow<saturate>
  } : (!ondsp.acc<storage = i48, frac = 30, signed, update_overflow = wrap>) -> i16
  return %result : i16
}

// The identity i64/frac30 destination materializes the raw accumulator: the
// frac is unchanged, so the export is exactly a widening sign extension and
// the harness can compare the folded i48 value itself.
func.func @q15_i48_wrap_raw_horizontal(
    %lhs: memref<?xi16>, %rhs: memref<?xi16>) -> i64 {
  %zero = ondsp.acc_zero
      : !ondsp.acc<storage = i48, frac = 30, signed, update_overflow = wrap>
  %acc = ondsp.reduce_mac %zero, %lhs, %rhs {
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    product = #ondsp.product<full>
  } : (!ondsp.acc<storage = i48, frac = 30, signed, update_overflow = wrap>, memref<?xi16>, memref<?xi16>) -> !ondsp.acc<storage = i48, frac = 30, signed, update_overflow = wrap>
  %result = ondsp.acc_export %acc {
    dst = #ondsp.fixed<signed, storage = i64, frac = 30>,
    rounding = #ondsp.rounding<nearest_even>,
    overflow = #ondsp.overflow<wrap>
  } : (!ondsp.acc<storage = i48, frac = 30, signed, update_overflow = wrap>) -> i64
  return %result : i64
}

func.func @q15_i48_wrap_raw_ordered(
    %lhs: memref<?xi16, strided<[?], offset: ?>>,
    %rhs: memref<?xi16, strided<[?], offset: ?>>) -> i64 {
  %zero = ondsp.acc_zero
      : !ondsp.acc<storage = i48, frac = 30, signed, update_overflow = wrap>
  %acc = ondsp.reduce_mac %zero, %lhs, %rhs {
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    product = #ondsp.product<full>
  } : (!ondsp.acc<storage = i48, frac = 30, signed, update_overflow = wrap>, memref<?xi16, strided<[?], offset: ?>>, memref<?xi16, strided<[?], offset: ?>>) -> !ondsp.acc<storage = i48, frac = 30, signed, update_overflow = wrap>
  %result = ondsp.acc_export %acc {
    dst = #ondsp.fixed<signed, storage = i64, frac = 30>,
    rounding = #ondsp.rounding<nearest_even>,
    overflow = #ondsp.overflow<wrap>
  } : (!ondsp.acc<storage = i48, frac = 30, signed, update_overflow = wrap>) -> i64
  return %result : i64
}
