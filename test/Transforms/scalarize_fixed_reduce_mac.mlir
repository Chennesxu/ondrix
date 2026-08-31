// RUN: ondrix-opt %s --scalarize-ondsp-fixed-reduce-mac --split-input-file | FileCheck %s
// RUN: ondrix-opt %s --scalarize-ondsp-fixed-reduce-mac=max-unrolled-terms=10 --split-input-file | FileCheck %s --check-prefix=FITS
// RUN: ondrix-opt %s --scalarize-ondsp-fixed-reduce-mac=max-unrolled-terms=9 --split-input-file | FileCheck %s --check-prefix=OVER

// A short static fixed reduction unrolls to its definitional straight-line
// ordered mac chain: the loop form would pay an index update and a branch
// per term.
// CHECK-LABEL: func.func @expands_static(
// CHECK-NOT: scf.for
// CHECK: %[[X0:.*]] = memref.load %{{.*}}[%c0] : memref<5xi16>
// CHECK: %[[C0:.*]] = memref.load %{{.*}}[%c0] : memref<5xi16>
// CHECK: %[[M0:.*]] = ondsp.mac %{{.*}}, %[[X0]], %[[C0]] {numeric = #ondsp.fixed<signed, storage = i16, frac = 15>, product = #ondsp.product<full>}
// CHECK-COUNT-4: ondsp.mac
// CHECK-NOT: scf.for
// CHECK: ondsp.acc_export
// CHECK-NOT: ondsp.reduce_mac
func.func @expands_static(%window: memref<5xi16>, %coefficients: memref<5xi16>) -> i16 {
  %z = ondsp.acc_zero : !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  %acc = ondsp.reduce_mac %z, %window, %coefficients {numeric = #ondsp.fixed<signed, storage = i16, frac = 15>, product = #ondsp.product<full>} : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>, memref<5xi16>, memref<5xi16>) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  %out = ondsp.acc_export %acc {dst = #ondsp.fixed<signed, storage = i16, frac = 15>, rounding = #ondsp.rounding<nearest_ties_positive>, overflow = #ondsp.overflow<saturate>} : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>) -> i16
  return %out : i16
}

// -----

// Dynamic lengths keep the same runtime equal-length proof the fixed scalar
// lowering inserts, then fold identically.
// CHECK-LABEL: func.func @expands_dynamic(
// CHECK: cf.assert %{{.*}}equal operand lengths
// CHECK: scf.for
// CHECK: ondsp.mac
// CHECK-NOT: ondsp.reduce_mac
func.func @expands_dynamic(%window: memref<?xi16>, %coefficients: memref<?xi16>) -> i16 {
  %z = ondsp.acc_zero : !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  %acc = ondsp.reduce_mac %z, %window, %coefficients {numeric = #ondsp.fixed<signed, storage = i16, frac = 15>, product = #ondsp.product<full>} : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>, memref<?xi16>, memref<?xi16>) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  %out = ondsp.acc_export %acc {dst = #ondsp.fixed<signed, storage = i16, frac = 15>, rounding = #ondsp.rounding<nearest_ties_positive>, overflow = #ondsp.overflow<saturate>} : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>) -> i16
  return %out : i16
}

// -----

// Past the unroll bound the static reduction keeps its ordered mac loop.
// CHECK-LABEL: func.func @expands_long_static(
// CHECK: scf.for %[[I:.*]] = %{{.*}} iter_args(%[[ACC:.*]] = %{{.*}}) -> (!ondsp.acc
// CHECK: ondsp.mac
// CHECK: scf.yield
// CHECK-NOT: ondsp.reduce_mac
func.func @expands_long_static(%window: memref<65xi16>, %coefficients: memref<65xi16>) -> i16 {
  %z = ondsp.acc_zero : !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  %acc = ondsp.reduce_mac %z, %window, %coefficients {numeric = #ondsp.fixed<signed, storage = i16, frac = 15>, product = #ondsp.product<full>} : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>, memref<65xi16>, memref<65xi16>) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  %out = ondsp.acc_export %acc {dst = #ondsp.fixed<signed, storage = i16, frac = 15>, rounding = #ondsp.rounding<nearest_ties_positive>, overflow = #ondsp.overflow<saturate>} : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>) -> i16
  return %out : i16
}

// -----

// Floating-point reductions belong to their own lowerings and stay.
// CHECK-LABEL: func.func @keeps_fp(
// CHECK: ondsp.reduce_mac
func.func @keeps_fp(%lhs: memref<8xf32>, %rhs: memref<8xf32>) -> f32 {
  %zero = arith.constant 0.0 : f32
  %r = ondsp.reduce_mac %zero, %lhs, %rhs {numeric = #ondsp.fp<format = f32, contract = off>} : (f32, memref<8xf32>, memref<8xf32>) -> f32
  return %r : f32
}

// -----

// The budget counts what a whole FUNCTION would emit, so a lowering that
// already replicated its reductions cannot multiply past it.
// FITS-LABEL: func.func @two_five_term_reductions(
// FITS-NOT: scf.for
// FITS-COUNT-10: ondsp.mac
// OVER-LABEL: func.func @two_five_term_reductions(
// OVER: scf.for
// OVER: scf.for
func.func @two_five_term_reductions(%window: memref<5xi16>, %coefficients: memref<5xi16>) -> (i16, i16) {
  %z0 = ondsp.acc_zero : !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  %a0 = ondsp.reduce_mac %z0, %window, %coefficients {numeric = #ondsp.fixed<signed, storage = i16, frac = 15>, product = #ondsp.product<full>} : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>, memref<5xi16>, memref<5xi16>) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  %o0 = ondsp.acc_export %a0 {dst = #ondsp.fixed<signed, storage = i16, frac = 15>, rounding = #ondsp.rounding<nearest_ties_positive>, overflow = #ondsp.overflow<saturate>} : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>) -> i16
  %z1 = ondsp.acc_zero : !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  %a1 = ondsp.reduce_mac %z1, %window, %coefficients {numeric = #ondsp.fixed<signed, storage = i16, frac = 15>, product = #ondsp.product<full>} : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>, memref<5xi16>, memref<5xi16>) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  %o1 = ondsp.acc_export %a1 {dst = #ondsp.fixed<signed, storage = i16, frac = 15>, rounding = #ondsp.rounding<nearest_ties_positive>, overflow = #ondsp.overflow<saturate>} : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>) -> i16
  return %o0, %o1 : i16, i16
}

// -----

// The budget prescan must ask the same question the rewrite asks. A reduction
// the rewrite will skip must not be priced, or it declines work never counted.
// CHECK-LABEL: func.func @budget_ignores_skipped_reductions
// CHECK-NOT: scf.for
// CHECK-COUNT-8: ondsp.mac
// The f32 reduction is eight terms the rewrite skips; only a prescan that
// counts them too would price this function at sixteen and refuse at nine.
// OVER-LABEL: func.func @budget_ignores_skipped_reductions
// OVER-NOT: scf.for
// OVER-COUNT-8: ondsp.mac
func.func @budget_ignores_skipped_reductions(
    %li: memref<8xi16>, %ri: memref<8xi16>,
    %lf: memref<8xf32>, %rf: memref<8xf32>) -> (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>, f32) {
  %s = ondsp.acc_zero : !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>
  %fixed = ondsp.reduce_mac %s, %li, %ri {numeric = #ondsp.fixed<signed, storage = i16, frac = 15>, product = #ondsp.product<full>} : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>, memref<8xi16>, memref<8xi16>) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>
  %z = arith.constant 0.0 : f32
  %fp = ondsp.reduce_mac %z, %lf, %rf {numeric = #ondsp.fp<format = f32, contract = off>} : (f32, memref<8xf32>, memref<8xf32>) -> f32
  return %fixed, %fp : !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>, f32
}
