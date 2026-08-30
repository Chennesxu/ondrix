// RUN: ondrix-opt %s --pass-pipeline='builtin.module(convert-ondrix-to-ondsp{preserve-bufferizable-reductions=true},empty-tensor-to-alloc-tensor,one-shot-bufferize{bufferize-function-boundaries=true allow-return-allocs=true function-boundary-type-conversion=identity-layout-map},cse,canonicalize)' > %t.bufferized.mlir
// RUN: ondrix-opt %t.bufferized.mlir --pass-pipeline='builtin.module(vectorize-ondsp-fp-filter-outputs{vector-width=8},vectorize-ondsp-fp-fast-memref-reduce{vector-width=8},vectorize-ondsp-fixed-decimate-outputs{vector-width=8},vectorize-ondsp-fixed-elementwise-updates{vector-width=8},vectorize-ondsp-constant-saturating-memref-reduce{vector-width=8 max-elements=64},vectorize-ondsp-fixed-memref-reduce{vector-width=8},parallelize-ondsp-fixed-wrap-vector-reduce,normalize-ondsp-fixed-vector-reduce)' > %t.once.mlir
// RUN: ondrix-opt %t.once.mlir --pass-pipeline='builtin.module(vectorize-ondsp-fp-filter-outputs{vector-width=8},vectorize-ondsp-fp-fast-memref-reduce{vector-width=8},vectorize-ondsp-fixed-decimate-outputs{vector-width=8},vectorize-ondsp-fixed-elementwise-updates{vector-width=8},vectorize-ondsp-constant-saturating-memref-reduce{vector-width=8 max-elements=64},vectorize-ondsp-fixed-memref-reduce{vector-width=8},parallelize-ondsp-fixed-wrap-vector-reduce,normalize-ondsp-fixed-vector-reduce)' > %t.twice.mlir
// RUN: diff %t.once.mlir %t.twice.mlir
// RUN: FileCheck %s --input-file=%t.once.mlir

// A second pass over an already-scheduled module must be a no-op: a site the
// stage served keeps the schedule it was given, and the ordered remainder a
// transform declined stays declined. The checks below keep the diff from
// passing vacuously on a stage that scheduled nothing.
// CHECK: vector.load
// CHECK: vector.reduction
func.func @f32_filter(%input: tensor<40xf32>, %coeffs: tensor<8xf32>,
                      %init: tensor<33xf32>) -> tensor<33xf32> {
  %result = ondrix.fir_filter %input, %coeffs, %init {
    boundary = #ondrix.fir_boundary<valid>,
    numeric = #ondsp.fp<format = f32, contract = fma>
  } : (tensor<40xf32>, tensor<8xf32>, tensor<33xf32>) -> tensor<33xf32>
  return %result : tensor<33xf32>
}

func.func @f32_dot_fast(%lhs: memref<?xf32>, %rhs: memref<?xf32>) -> f32 {
  %zero = arith.constant 0.0 : f32
  %r = ondsp.reduce_mac %zero, %lhs, %rhs {numeric = #ondsp.fp<format = f32, contract = fast>} : (f32, memref<?xf32>, memref<?xf32>) -> f32
  return %r : f32
}

func.func @q15_filter(%input: tensor<40xi16>, %coeffs: tensor<8xi16>,
                      %init: tensor<33xi16>) -> tensor<33xi16> {
  %result = ondrix.fir_filter %input, %coeffs, %init {
    boundary = #ondrix.fir_boundary<valid>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    product = #ondsp.product<full>,
    accumulator = !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>,
    dst = #ondsp.fixed<signed, storage = i16, frac = 15>,
    rounding = #ondsp.rounding<nearest_even>,
    overflow = #ondsp.overflow<saturate>
  } : (tensor<40xi16>, tensor<8xi16>, tensor<33xi16>) -> tensor<33xi16>
  return %result : tensor<33xi16>
}

func.func @q15_lms(%x: tensor<40xi16>, %d: tensor<40xi16>, %w: tensor<11xi16>)
    -> (tensor<40xi16>, tensor<11xi16>) {
  %error, %adapted = ondrix.lms %x, %d, %w {
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    rounding = #ondsp.rounding<nearest_even>,
    step_size = 4096 : i64
  } : (tensor<40xi16>, tensor<40xi16>, tensor<11xi16>) -> (tensor<40xi16>, tensor<11xi16>)
  return %error, %adapted : tensor<40xi16>, tensor<11xi16>
}
