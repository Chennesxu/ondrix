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
// Both kernels declare the same contract and differ only in the memref
// layout. The identity-layout operands admit the wrap horizontal path; the
// dynamic-stride operands fall back to the ordered per-element fold. The
// final accumulator leaves through one declared Q15 boundary.

// PLAN-LABEL: func.func @q15_i48_wrap_horizontal
// PLAN: vector.reduction <add>
// PLAN: ondsp.acc_add_term
// PLAN-SAME: !ondsp.acc<storage = i48, frac = 30, signed, update_overflow = wrap>

// PLAN-LABEL: func.func @q15_i48_wrap_ordered
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
