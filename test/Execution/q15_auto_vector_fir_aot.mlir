// RUN: ondrix-opt %s --convert-ondrix-to-ondsp --vectorize-ondsp-q15-memref-reduce="vector-width=8" --parallelize-ondsp-q15-wrap-vector-reduce --normalize-ondsp-q15-vector-reduce --convert-ondsp-fixed-to-scalar --convert-scf-to-cf --convert-vector-to-llvm --finalize-memref-to-llvm --convert-arith-to-llvm --convert-cf-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts > %t.mlir
// RUN: ondrix-translate %t.mlir --mlir-to-llvmir > %t.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.ll -o %t.o
// RUN: cc %S/Inputs/q15_auto_vector_fir_aot.c %t.o -o %t
// RUN: %t
// RUN: cc %S/Inputs/q15_auto_vector_fir_mismatch.c %t.o -o %t.mismatch
// RUN: not --crash %t.mismatch
// RUN: llc -O2 -mtriple=x86_64-unknown-linux-gnu -mattr=+avx2 -filetype=asm %t.ll -o %t.s
// RUN: FileCheck %s --check-prefix=AVX2 < %t.s

// AVX2-LABEL: q15_auto_vector_saturate:
// AVX2: vpmulld
// AVX2-LABEL: q15_auto_vector_wrap:
// AVX2: vpmulld
// AVX2: vpaddq
// AVX2-LABEL: q15_seeded_vector_wrap:
// AVX2: vpmulld
// AVX2: vpaddq
// AVX2-LABEL: q15_offset_vector_saturate:
// AVX2: vpmulld

func.func @q15_auto_vector_saturate(
    %input: memref<?xi16>, %coeffs: memref<?xi16>) -> i16 {
  %acc = ondrix.fir %input, %coeffs {
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    product = #ondsp.product<full>
  } : (memref<?xi16>, memref<?xi16>) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  %result = ondsp.acc_export %acc {
    dst = #ondsp.fixed<signed, storage = i16, frac = 15>,
    rounding = #ondsp.rounding<toward_negative>,
    overflow = #ondsp.overflow<wrap>
  } : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>) -> i16
  return %result : i16
}

func.func @q15_auto_vector_wrap(
    %lhs: memref<?xi16>, %rhs: memref<?xi16>) -> i16 {
  %acc = ondrix.dot %lhs, %rhs {
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    product = #ondsp.product<full>
  } : (memref<?xi16>, memref<?xi16>) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>
  %result = ondsp.acc_export %acc {
    dst = #ondsp.fixed<signed, storage = i16, frac = 15>,
    rounding = #ondsp.rounding<toward_negative>,
    overflow = #ondsp.overflow<wrap>
  } : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>) -> i16
  return %result : i16
}

func.func @q15_seeded_vector_wrap(
    %seed: i16, %lhs: memref<?xi16>, %rhs: memref<?xi16>) -> i16 {
  %initial = ondsp.acc_import %seed {
    src = #ondsp.fixed<signed, storage = i16, frac = 15>
  } : (i16) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>
  %acc = ondsp.reduce_mac %initial, %lhs, %rhs {
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    product = #ondsp.product<full>
  } : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>, memref<?xi16>, memref<?xi16>) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>
  %result = ondsp.acc_export %acc {
    dst = #ondsp.fixed<signed, storage = i16, frac = 15>,
    rounding = #ondsp.rounding<toward_negative>,
    overflow = #ondsp.overflow<wrap>
  } : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>) -> i16
  return %result : i16
}

func.func @q15_offset_vector_saturate(
    %input: memref<?xi16, strided<[1], offset: ?>>,
    %coeffs: memref<?xi16, strided<[1], offset: ?>>) -> i16 {
  %acc = ondrix.fir %input, %coeffs {
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    product = #ondsp.product<full>
  } : (memref<?xi16, strided<[1], offset: ?>>, memref<?xi16, strided<[1], offset: ?>>) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  %result = ondsp.acc_export %acc {
    dst = #ondsp.fixed<signed, storage = i16, frac = 15>,
    rounding = #ondsp.rounding<toward_negative>,
    overflow = #ondsp.overflow<wrap>
  } : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>) -> i16
  return %result : i16
}

func.func @q15_scalar_fallback_saturate(
    %input: memref<?xi16, strided<[?], offset: ?>>,
    %coeffs: memref<?xi16, strided<[?], offset: ?>>) -> i16 {
  %acc = ondrix.fir %input, %coeffs {
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    product = #ondsp.product<full>
  } : (memref<?xi16, strided<[?], offset: ?>>, memref<?xi16, strided<[?], offset: ?>>) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  %result = ondsp.acc_export %acc {
    dst = #ondsp.fixed<signed, storage = i16, frac = 15>,
    rounding = #ondsp.rounding<toward_negative>,
    overflow = #ondsp.overflow<wrap>
  } : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>) -> i16
  return %result : i16
}

func.func @q15_scalar_fallback_wrap(
    %lhs: memref<?xi16, strided<[?], offset: ?>>,
    %rhs: memref<?xi16, strided<[?], offset: ?>>) -> i16 {
  %acc = ondrix.dot %lhs, %rhs {
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    product = #ondsp.product<full>
  } : (memref<?xi16, strided<[?], offset: ?>>, memref<?xi16, strided<[?], offset: ?>>) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>
  %result = ondsp.acc_export %acc {
    dst = #ondsp.fixed<signed, storage = i16, frac = 15>,
    rounding = #ondsp.rounding<toward_negative>,
    overflow = #ondsp.overflow<wrap>
  } : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>) -> i16
  return %result : i16
}
