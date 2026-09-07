// RUN: ondrix-opt %s --unroll-ondsp-fp-ordered-reduce="max-straight-line-terms=8 max-unrolled-terms=16" --split-input-file | FileCheck %s

// An ordered reduction becomes the chain its contract denotes: product then
// fold, in index order, onto the same initial accumulator.
// CHECK-LABEL: func.func @straightens_ordered_reduction(
// CHECK-NOT: scf.for
// CHECK-COUNT-4: arith.mulf
// CHECK-NOT: ondsp.reduce_mac
func.func @straightens_ordered_reduction(%a: memref<4xf32>, %b: memref<4xf32>) -> f32 {
  %seed = arith.constant 0.000000e+00 : f32
  %r = ondsp.reduce_mac %seed, %a, %b {numeric = #ondsp.fp<format = f32, contract = off>} : (f32, memref<4xf32>, memref<4xf32>) -> f32
  return %r : f32
}

// -----

// The fold is onto the accumulator, not onto the product: the operand order
// the scalar lowering's own loop body uses, so the two forms agree bitwise.
// CHECK-LABEL: func.func @folds_onto_the_accumulator(
// CHECK: %[[P0:.*]] = arith.mulf
// CHECK: %[[S0:.*]] = arith.addf %{{.*}}, %[[P0]]
// CHECK: %[[P1:.*]] = arith.mulf
// CHECK: arith.addf %[[S0]], %[[P1]]
func.func @folds_onto_the_accumulator(%a: memref<2xf32>, %b: memref<2xf32>, %seed: f32) -> f32 {
  %r = ondsp.reduce_mac %seed, %a, %b {numeric = #ondsp.fp<format = f32, contract = off>} : (f32, memref<2xf32>, memref<2xf32>) -> f32
  return %r : f32
}

// -----

// A fast site carries spent-permission records this pass must not replicate,
// and the fast reduction pass owns its shape.
// CHECK-LABEL: func.func @keeps_fast_contract(
// CHECK: ondsp.reduce_mac
func.func @keeps_fast_contract(%a: memref<4xf32>, %b: memref<4xf32>) -> f32 {
  %seed = arith.constant 0.000000e+00 : f32
  %r = ondsp.reduce_mac %seed, %a, %b {numeric = #ondsp.fp<format = f32, contract = fast>} : (f32, memref<4xf32>, memref<4xf32>) -> f32
  return %r : f32
}

// -----

// Past the per-site cap the loop form is kept.
// CHECK-LABEL: func.func @keeps_long_reduction(
// CHECK: ondsp.reduce_mac
func.func @keeps_long_reduction(%a: memref<16xf32>, %b: memref<16xf32>) -> f32 {
  %seed = arith.constant 0.000000e+00 : f32
  %r = ondsp.reduce_mac %seed, %a, %b {numeric = #ondsp.fp<format = f32, contract = off>} : (f32, memref<16xf32>, memref<16xf32>) -> f32
  return %r : f32
}

// -----

// The tap loop the FIR family's lowering leaves behind: one f32 accumulator
// over a compile-time trip count.
// CHECK-LABEL: func.func @unrolls_f32_tap_loop(
// CHECK-NOT: scf.for
// CHECK-COUNT-4: arith.addf
func.func @unrolls_f32_tap_loop(%x: memref<64xf32>, %c: memref<4xf32>) -> f32 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %seed = arith.constant 0.000000e+00 : f32
  %acc = scf.for %tap = %c0 to %c4 step %c1 iter_args(%chain = %seed) -> (f32) {
    %sample = memref.load %x[%tap] : memref<64xf32>
    %coefficient = memref.load %c[%tap] : memref<4xf32>
    %product = arith.mulf %sample, %coefficient : f32
    %next = arith.addf %chain, %product : f32
    scf.yield %next : f32
  }
  return %acc : f32
}

// -----

// A dynamic bound has no compile-time trip count to expand.
// CHECK-LABEL: func.func @keeps_dynamic_tap_loop(
// CHECK: scf.for
func.func @keeps_dynamic_tap_loop(%x: memref<64xf32>, %c: memref<?xf32>, %taps: index) -> f32 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %seed = arith.constant 0.000000e+00 : f32
  %acc = scf.for %tap = %c0 to %taps step %c1 iter_args(%chain = %seed) -> (f32) {
    %sample = memref.load %x[%tap] : memref<64xf32>
    %coefficient = memref.load %c[%tap] : memref<?xf32>
    %product = arith.mulf %sample, %coefficient : f32
    %next = arith.addf %chain, %product : f32
    scf.yield %next : f32
  }
  return %acc : f32
}

// -----

// The budget counts the function, not the site: three eight-term reductions
// pass the per-site cap and together exceed the function's sixteen, so every
// one of them keeps its loop rather than the function mixing both shapes.
// CHECK-LABEL: func.func @budget_is_per_function(
// CHECK-COUNT-3: ondsp.reduce_mac
func.func @budget_is_per_function(%a: memref<8xf32>, %b: memref<8xf32>) -> f32 {
  %seed = arith.constant 0.000000e+00 : f32
  %r0 = ondsp.reduce_mac %seed, %a, %b {numeric = #ondsp.fp<format = f32, contract = off>} : (f32, memref<8xf32>, memref<8xf32>) -> f32
  %r1 = ondsp.reduce_mac %r0, %a, %b {numeric = #ondsp.fp<format = f32, contract = off>} : (f32, memref<8xf32>, memref<8xf32>) -> f32
  %r2 = ondsp.reduce_mac %r1, %a, %b {numeric = #ondsp.fp<format = f32, contract = off>} : (f32, memref<8xf32>, memref<8xf32>) -> f32
  return %r2 : f32
}

// -----

// RUN: ondrix-opt %s --unroll-ondsp-fp-ordered-reduce="vector-width=4 max-straight-line-terms=8" \
// RUN:   --split-input-file | FileCheck %s --check-prefix=LANES

// With lanes to fill, the reduction is left to the lane-blocked scalar
// lowering while the accumulator loop, which no lane stage claims, is still
// taken.
// LANES-LABEL: func.func @lanes_keep_the_reduction(
// LANES: ondsp.reduce_mac
// LANES-NOT: scf.for
func.func @lanes_keep_the_reduction(%a: memref<4xf32>, %b: memref<4xf32>, %x: memref<64xf32>,
                                    %c: memref<4xf32>) -> (f32, f32) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %seed = arith.constant 0.000000e+00 : f32
  %r = ondsp.reduce_mac %seed, %a, %b {numeric = #ondsp.fp<format = f32, contract = off>} : (f32, memref<4xf32>, memref<4xf32>) -> f32
  %acc = scf.for %tap = %c0 to %c4 step %c1 iter_args(%chain = %seed) -> (f32) {
    %sample = memref.load %x[%tap] : memref<64xf32>
    %coefficient = memref.load %c[%tap] : memref<4xf32>
    %product = arith.mulf %sample, %coefficient : f32
    %next = arith.addf %chain, %product : f32
    scf.yield %next : f32
  }
  return %r, %acc : f32, f32
}
