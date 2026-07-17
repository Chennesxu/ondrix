// RUN: ondrix-opt %s --convert-ondrix-to-ondsp --vectorize-ondsp-fixed-memref-reduce="vector-width=4" --parallelize-ondsp-fixed-wrap-vector-reduce --normalize-ondsp-fixed-vector-reduce > %t.vector.mlir
// RUN: FileCheck %s --check-prefix=VECTOR < %t.vector.mlir
// RUN: ondrix-opt %t.vector.mlir --convert-ondsp-fixed-to-scalar --convert-scf-to-cf --convert-vector-to-llvm --finalize-memref-to-llvm --convert-arith-to-llvm --convert-cf-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts > %t.mlir
// RUN: ondrix-translate %t.mlir --mlir-to-llvmir > %t.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.ll -o %t.o
// RUN: cc %S/Inputs/q31_auto_vector_aot.c %t.o -o %t
// RUN: %t
// RUN: cc %S/Inputs/q31_auto_vector_mismatch.c %t.o -o %t.mismatch
// RUN: not --crash %t.mismatch

func.func @q31_vector_full_raw_saturate(
    %seed: i32, %lhs: memref<?xi32>, %rhs: memref<?xi32>) -> i64 {
  %initial = ondsp.acc_import %seed {
    src = #ondsp.fixed<signed, storage = i32, frac = 31>
  } : (i32) -> !ondsp.acc<storage = i64, frac = 62, signed, update_overflow = saturate>
  %acc = ondsp.reduce_mac %initial, %lhs, %rhs {
    numeric = #ondsp.fixed<signed, storage = i32, frac = 31>,
    product = #ondsp.product<full>
  } : (!ondsp.acc<storage = i64, frac = 62, signed, update_overflow = saturate>, memref<?xi32>, memref<?xi32>) -> !ondsp.acc<storage = i64, frac = 62, signed, update_overflow = saturate>
  %raw = builtin.unrealized_conversion_cast %acc : !ondsp.acc<storage = i64, frac = 62, signed, update_overflow = saturate> to i64
  return %raw : i64
}

func.func @q31_vector_full_raw_wrap(
    %seed: i32, %lhs: memref<?xi32>, %rhs: memref<?xi32>) -> i64 {
  %initial = ondsp.acc_import %seed {
    src = #ondsp.fixed<signed, storage = i32, frac = 31>
  } : (i32) -> !ondsp.acc<storage = i64, frac = 62, signed, update_overflow = wrap>
  %acc = ondsp.reduce_mac %initial, %lhs, %rhs {
    numeric = #ondsp.fixed<signed, storage = i32, frac = 31>,
    product = #ondsp.product<full>
  } : (!ondsp.acc<storage = i64, frac = 62, signed, update_overflow = wrap>, memref<?xi32>, memref<?xi32>) -> !ondsp.acc<storage = i64, frac = 62, signed, update_overflow = wrap>
  %raw = builtin.unrealized_conversion_cast %acc : !ondsp.acc<storage = i64, frac = 62, signed, update_overflow = wrap> to i64
  return %raw : i64
}

func.func @q31_vector_high_raw_saturate(
    %seed: i32, %lhs: memref<?xi32>, %rhs: memref<?xi32>) -> i64 {
  %initial = ondsp.acc_import %seed {
    src = #ondsp.fixed<signed, storage = i32, frac = 30>
  } : (i32) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  %acc = ondsp.reduce_mac %initial, %lhs, %rhs {
    numeric = #ondsp.fixed<signed, storage = i32, frac = 31>,
    product = #ondsp.product<high_raw>
  } : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>, memref<?xi32>, memref<?xi32>) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  %raw = builtin.unrealized_conversion_cast %acc : !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate> to i40
  %wide = arith.extsi %raw : i40 to i64
  return %wide : i64
}

func.func @q31_vector_high_raw_wrap(
    %seed: i32, %lhs: memref<?xi32>, %rhs: memref<?xi32>) -> i64 {
  %initial = ondsp.acc_import %seed {
    src = #ondsp.fixed<signed, storage = i32, frac = 30>
  } : (i32) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>
  %acc = ondsp.reduce_mac %initial, %lhs, %rhs {
    numeric = #ondsp.fixed<signed, storage = i32, frac = 31>,
    product = #ondsp.product<high_raw>
  } : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>, memref<?xi32>, memref<?xi32>) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>
  %raw = builtin.unrealized_conversion_cast %acc : !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap> to i40
  %wide = arith.extsi %raw : i40 to i64
  return %wide : i64
}

func.func @q31_vector_high_raw_offset_saturate(
    %seed: i32,
    %lhs: memref<?xi32, strided<[1], offset: ?>>,
    %rhs: memref<?xi32, strided<[1], offset: ?>>) -> i64 {
  %initial = ondsp.acc_import %seed {
    src = #ondsp.fixed<signed, storage = i32, frac = 30>
  } : (i32) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  %acc = ondsp.reduce_mac %initial, %lhs, %rhs {
    numeric = #ondsp.fixed<signed, storage = i32, frac = 31>,
    product = #ondsp.product<high_raw>
  } : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>, memref<?xi32, strided<[1], offset: ?>>, memref<?xi32, strided<[1], offset: ?>>) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  %raw = builtin.unrealized_conversion_cast %acc : !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate> to i40
  %wide = arith.extsi %raw : i40 to i64
  return %wide : i64
}

func.func @q31_scalar_full_raw_saturate(
    %seed: i32,
    %lhs: memref<?xi32, strided<[?], offset: ?>>,
    %rhs: memref<?xi32, strided<[?], offset: ?>>) -> i64 {
  %initial = ondsp.acc_import %seed {
    src = #ondsp.fixed<signed, storage = i32, frac = 31>
  } : (i32) -> !ondsp.acc<storage = i64, frac = 62, signed, update_overflow = saturate>
  %acc = ondsp.reduce_mac %initial, %lhs, %rhs {
    numeric = #ondsp.fixed<signed, storage = i32, frac = 31>,
    product = #ondsp.product<full>
  } : (!ondsp.acc<storage = i64, frac = 62, signed, update_overflow = saturate>, memref<?xi32, strided<[?], offset: ?>>, memref<?xi32, strided<[?], offset: ?>>) -> !ondsp.acc<storage = i64, frac = 62, signed, update_overflow = saturate>
  %raw = builtin.unrealized_conversion_cast %acc : !ondsp.acc<storage = i64, frac = 62, signed, update_overflow = saturate> to i64
  return %raw : i64
}

func.func @q31_scalar_full_raw_wrap(
    %seed: i32,
    %lhs: memref<?xi32, strided<[?], offset: ?>>,
    %rhs: memref<?xi32, strided<[?], offset: ?>>) -> i64 {
  %initial = ondsp.acc_import %seed {
    src = #ondsp.fixed<signed, storage = i32, frac = 31>
  } : (i32) -> !ondsp.acc<storage = i64, frac = 62, signed, update_overflow = wrap>
  %acc = ondsp.reduce_mac %initial, %lhs, %rhs {
    numeric = #ondsp.fixed<signed, storage = i32, frac = 31>,
    product = #ondsp.product<full>
  } : (!ondsp.acc<storage = i64, frac = 62, signed, update_overflow = wrap>, memref<?xi32, strided<[?], offset: ?>>, memref<?xi32, strided<[?], offset: ?>>) -> !ondsp.acc<storage = i64, frac = 62, signed, update_overflow = wrap>
  %raw = builtin.unrealized_conversion_cast %acc : !ondsp.acc<storage = i64, frac = 62, signed, update_overflow = wrap> to i64
  return %raw : i64
}

func.func @q31_scalar_high_raw_saturate(
    %seed: i32,
    %lhs: memref<?xi32, strided<[?], offset: ?>>,
    %rhs: memref<?xi32, strided<[?], offset: ?>>) -> i64 {
  %initial = ondsp.acc_import %seed {
    src = #ondsp.fixed<signed, storage = i32, frac = 30>
  } : (i32) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  %acc = ondsp.reduce_mac %initial, %lhs, %rhs {
    numeric = #ondsp.fixed<signed, storage = i32, frac = 31>,
    product = #ondsp.product<high_raw>
  } : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>, memref<?xi32, strided<[?], offset: ?>>, memref<?xi32, strided<[?], offset: ?>>) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  %raw = builtin.unrealized_conversion_cast %acc : !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate> to i40
  %wide = arith.extsi %raw : i40 to i64
  return %wide : i64
}

func.func @q31_scalar_high_raw_wrap(
    %seed: i32,
    %lhs: memref<?xi32, strided<[?], offset: ?>>,
    %rhs: memref<?xi32, strided<[?], offset: ?>>) -> i64 {
  %initial = ondsp.acc_import %seed {
    src = #ondsp.fixed<signed, storage = i32, frac = 30>
  } : (i32) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>
  %acc = ondsp.reduce_mac %initial, %lhs, %rhs {
    numeric = #ondsp.fixed<signed, storage = i32, frac = 31>,
    product = #ondsp.product<high_raw>
  } : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>, memref<?xi32, strided<[?], offset: ?>>, memref<?xi32, strided<[?], offset: ?>>) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>
  %raw = builtin.unrealized_conversion_cast %acc : !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap> to i40
  %wide = arith.extsi %raw : i40 to i64
  return %wide : i64
}

func.func @q31_vector_full_export(
    %lhs: memref<?xi32>, %rhs: memref<?xi32>) -> i32 {
  %acc = ondrix.dot %lhs, %rhs {
    numeric = #ondsp.fixed<signed, storage = i32, frac = 31>,
    product = #ondsp.product<full>
  } : (memref<?xi32>, memref<?xi32>) -> !ondsp.acc<storage = i64, frac = 62, signed, update_overflow = saturate>
  %result = ondsp.acc_export %acc {
    dst = #ondsp.fixed<signed, storage = i32, frac = 31>,
    rounding = #ondsp.rounding<nearest_even>,
    overflow = #ondsp.overflow<saturate>
  } : (!ondsp.acc<storage = i64, frac = 62, signed, update_overflow = saturate>) -> i32
  return %result : i32
}

func.func @q31_vector_high_raw_export_q30(
    %lhs: memref<?xi32, strided<[1], offset: ?>>,
    %rhs: memref<?xi32, strided<[1], offset: ?>>) -> i32 {
  %acc = ondrix.fir %lhs, %rhs {
    numeric = #ondsp.fixed<signed, storage = i32, frac = 31>,
    product = #ondsp.product<high_raw>
  } : (memref<?xi32, strided<[1], offset: ?>>, memref<?xi32, strided<[1], offset: ?>>) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  %result = ondsp.acc_export %acc {
    dst = #ondsp.fixed<signed, storage = i32, frac = 30>,
    rounding = #ondsp.rounding<toward_negative>,
    overflow = #ondsp.overflow<saturate>
  } : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>) -> i32
  return %result : i32
}

// VECTOR-LABEL: func.func @q31_vector_full_raw_saturate
// VECTOR: vector.load {{.*}} : memref<?xi32>, vector<4xi32>
// VECTOR: arith.muli {{.*}} : vector<4xi64>
// VECTOR: vector.extract {{.*}}[0] : vector<4xi64>

// VECTOR-LABEL: func.func @q31_vector_full_raw_wrap
// VECTOR: vector.load {{.*}} : memref<?xi32>, vector<4xi32>
// VECTOR: vector.reduction <add>, {{.*}} : vector<4xi64> into i64

// VECTOR-LABEL: func.func @q31_vector_high_raw_saturate
// VECTOR: arith.constant dense<32> : vector<4xi64>
// VECTOR: arith.shrsi {{.*}} : vector<4xi64>
// VECTOR: vector.extract {{.*}}[0] : vector<4xi32>

// VECTOR-LABEL: func.func @q31_vector_high_raw_wrap
// VECTOR: arith.constant dense<32> : vector<4xi64>
// VECTOR: vector.reduction <add>, {{.*}} : vector<4xi64> into i64

// VECTOR-LABEL: func.func @q31_scalar_full_raw_saturate
// VECTOR-NOT: vector.load
// VECTOR: ondsp.reduce_mac
