// RUN: ondrix-opt %s --convert-ondrix-to-ondsp > %t.ondsp.mlir
// RUN: ondrix-opt %t.ondsp.mlir --vectorize-ondsp-constant-saturating-memref-reduce="vector-width=4 max-elements=64 proof-trace-output=%t.proof.json" > %t.proven.mlir
// RUN: FileCheck %s --check-prefix=PROVEN < %t.proven.mlir
// RUN: FileCheck %s --check-prefix=TRACE --input-file=%t.proof.json
// RUN: ondrix-opt %t.ondsp.mlir --verify-ondsp-constant-reassociation-proof-trace="proof-trace-input=%t.proof.json max-elements=64" > /dev/null
// RUN: ondrix-opt %t.proven.mlir --vectorize-ondsp-fixed-memref-reduce="vector-width=4" --parallelize-ondsp-fixed-wrap-vector-reduce --normalize-ondsp-fixed-vector-reduce --convert-ondsp-fixed-to-scalar --convert-scf-to-cf --convert-vector-to-llvm --finalize-memref-to-llvm --convert-arith-to-llvm --convert-cf-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts > %t.mlir
// RUN: ondrix-translate %t.mlir --mlir-to-llvmir > %t.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.ll -o %t.o
// RUN: cc %S/Inputs/constant_saturating_vector_aot.c %t.o -o %t
// RUN: %t
// RUN: llc -O2 -mtriple=x86_64-unknown-linux-gnu -mattr=+avx2 -filetype=asm %t.ll -o %t.s
// RUN: FileCheck %s --check-prefix=AVX2 < %t.s

// PROVEN-LABEL: func.func @q15_proven_vector
// PROVEN: vector.reduction <add>, {{.*}} : vector<4xi64> into i64
// PROVEN-NOT: ondsp.reduce_mac
// PROVEN-LABEL: func.func @q15_scalar_reference
// PROVEN: ondsp.reduce_mac
// PROVEN-NOT: vector.reduction
// PROVEN-LABEL: func.func @q15_proven_offset
// PROVEN: vector.reduction <add>, {{.*}} : vector<4xi64> into i64
// PROVEN-NOT: ondsp.reduce_mac
// PROVEN-LABEL: func.func @q31_proven_vector
// PROVEN: vector.reduction <add>, {{.*}} : vector<4xi64> into i64
// PROVEN-NOT: ondsp.reduce_mac
// PROVEN-LABEL: func.func @q31_scalar_reference
// PROVEN: ondsp.reduce_mac
// PROVEN-NOT: vector.reduction

// TRACE-DAG: "candidate_reduction_count":{{ *}}5
// TRACE-DAG: "subject_ordinal":{{ *}}0
// TRACE-DAG: "subject_ordinal":{{ *}}2
// TRACE-DAG: "subject_ordinal":{{ *}}3

// AVX2-LABEL: q15_proven_vector:
// AVX2: vpmaddwd
// AVX2: vpaddq

memref.global "private" constant @q15_coefficients : memref<17xi16> =
  dense<[1, -2, 3, -4, 5, -6, 7, -8, 9, -10, 11, -12, 13, -14, 15, -16, 17]>
memref.global "private" constant @q31_coefficients : memref<5xi32> =
  dense<[1, -2, 3, -4, 5]>

func.func @q15_proven_vector(%input: memref<17xi16>) -> i16 {
  %coefficients = memref.get_global @q15_coefficients : memref<17xi16>
  %accumulator = ondrix.dot %input, %coefficients {
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    product = #ondsp.product<full>
  } : (memref<17xi16>, memref<17xi16>)
      -> !ondsp.acc<storage = i40, frac = 30, signed,
                    update_overflow = saturate>
  %result = ondsp.acc_export %accumulator {
    dst = #ondsp.fixed<signed, storage = i16, frac = 15>,
    rounding = #ondsp.rounding<toward_negative>,
    overflow = #ondsp.overflow<wrap>
  } : (!ondsp.acc<storage = i40, frac = 30, signed,
                       update_overflow = saturate>) -> i16
  return %result : i16
}

func.func @q15_scalar_reference(
    %input: memref<17xi16, strided<[?], offset: ?>>) -> i16 {
  %coefficients = memref.get_global @q15_coefficients : memref<17xi16>
  %accumulator = ondrix.fir %input, %coefficients {
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    product = #ondsp.product<full>
  } : (memref<17xi16, strided<[?], offset: ?>>, memref<17xi16>)
      -> !ondsp.acc<storage = i40, frac = 30, signed,
                    update_overflow = saturate>
  %result = ondsp.acc_export %accumulator {
    dst = #ondsp.fixed<signed, storage = i16, frac = 15>,
    rounding = #ondsp.rounding<toward_negative>,
    overflow = #ondsp.overflow<wrap>
  } : (!ondsp.acc<storage = i40, frac = 30, signed,
                       update_overflow = saturate>) -> i16
  return %result : i16
}

func.func @q15_proven_offset(
    %input: memref<17xi16, strided<[1], offset: ?>>) -> i16 {
  %coefficients = memref.get_global @q15_coefficients : memref<17xi16>
  %accumulator = ondrix.fir %input, %coefficients {
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    product = #ondsp.product<full>
  } : (memref<17xi16, strided<[1], offset: ?>>, memref<17xi16>)
      -> !ondsp.acc<storage = i40, frac = 30, signed,
                    update_overflow = saturate>
  %result = ondsp.acc_export %accumulator {
    dst = #ondsp.fixed<signed, storage = i16, frac = 15>,
    rounding = #ondsp.rounding<toward_negative>,
    overflow = #ondsp.overflow<wrap>
  } : (!ondsp.acc<storage = i40, frac = 30, signed,
                       update_overflow = saturate>) -> i16
  return %result : i16
}

func.func @q31_proven_vector(%input: memref<5xi32>) -> i32 {
  %coefficients = memref.get_global @q31_coefficients : memref<5xi32>
  %accumulator = ondrix.dot %input, %coefficients {
    numeric = #ondsp.fixed<signed, storage = i32, frac = 31>,
    product = #ondsp.product<full>
  } : (memref<5xi32>, memref<5xi32>)
      -> !ondsp.acc<storage = i64, frac = 62, signed,
                    update_overflow = saturate>
  %result = ondsp.acc_export %accumulator {
    dst = #ondsp.fixed<signed, storage = i32, frac = 31>,
    rounding = #ondsp.rounding<toward_negative>,
    overflow = #ondsp.overflow<wrap>
  } : (!ondsp.acc<storage = i64, frac = 62, signed,
                       update_overflow = saturate>) -> i32
  return %result : i32
}

func.func @q31_scalar_reference(
    %input: memref<5xi32, strided<[?], offset: ?>>) -> i32 {
  %coefficients = memref.get_global @q31_coefficients : memref<5xi32>
  %accumulator = ondrix.fir %input, %coefficients {
    numeric = #ondsp.fixed<signed, storage = i32, frac = 31>,
    product = #ondsp.product<full>
  } : (memref<5xi32, strided<[?], offset: ?>>, memref<5xi32>)
      -> !ondsp.acc<storage = i64, frac = 62, signed,
                    update_overflow = saturate>
  %result = ondsp.acc_export %accumulator {
    dst = #ondsp.fixed<signed, storage = i32, frac = 31>,
    rounding = #ondsp.rounding<toward_negative>,
    overflow = #ondsp.overflow<wrap>
  } : (!ondsp.acc<storage = i64, frac = 62, signed,
                       update_overflow = saturate>) -> i32
  return %result : i32
}
