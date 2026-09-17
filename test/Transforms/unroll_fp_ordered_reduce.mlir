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

// -----

// The interleaved chain loop a fast reduction leaves behind: several f32
// accumulators, and bounds the producing pass built from constants that
// nothing canonicalizes before this pass runs.
// CHECK-LABEL: func.func @unrolls_multi_chain_loop(
// CHECK-NOT: scf.for
// CHECK-COUNT-8: arith.addf
func.func @unrolls_multi_chain_loop(%x: memref<64xf32>, %seed: f32) -> f32 {
  %c0 = arith.constant 0 : index
  %c2 = arith.constant 2 : index
  %c8 = arith.constant 8 : index
  %end = arith.subi %c8, %c0 : index
  %rem = arith.remui %end, %c2 : index
  %stop = arith.subi %end, %rem : index
  %r:2 = scf.for %i = %c0 to %stop step %c2 iter_args(%p = %seed, %q = %seed) -> (f32, f32) {
    %a = memref.load %x[%i] : memref<64xf32>
    %b = memref.load %x[%i] : memref<64xf32>
    %pn = arith.addf %p, %a : f32
    %qn = arith.addf %q, %b : f32
    scf.yield %pn, %qn : f32, f32
  }
  %s = arith.addf %r#0, %r#1 : f32
  return %s : f32
}

// -----

// RUN: ondrix-opt %s --unroll-ondsp-fp-ordered-reduce="max-straight-line-terms=8 max-block-terms=6" \
// RUN:   --split-input-file | FileCheck %s --check-prefix=BLOCK

// A loop that stores an output every trip is a sample recurrence: it is replayed
// in blocks (here two trips of three terms), never as one straight line.
// BLOCK-LABEL: func.func @blocks_sample_recurrence(
// BLOCK: %[[STEP:.*]] = arith.constant 2 : index
// BLOCK: scf.for %{{.*}} = %{{.*}} to %{{.*}} step %[[STEP]] iter_args(%[[D1:.*]] = %{{.*}}, %[[D2:.*]] = %{{.*}})
// BLOCK-COUNT-2: memref.store
// BLOCK: scf.yield
// BLOCK-NOT: memref.store
func.func @blocks_sample_recurrence(%x: memref<8xf32>, %y: memref<8xf32>, %b: f32, %seed: f32) -> f32 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %r:2 = scf.for %i = %c0 to %c8 step %c1 iter_args(%d1 = %seed, %d2 = %seed) -> (f32, f32) {
    %in = memref.load %x[%i] : memref<8xf32>
    %w = math.fma %d1, %b, %in : f32
    %wn = math.fma %d2, %b, %w : f32
    %out = arith.addf %wn, %d1 : f32
    memref.store %out, %y[%i] : memref<8xf32>
    scf.yield %wn, %d1 : f32, f32
  }
  return %r#0 : f32
}

// -----

// A bufferized window sum stamps its declaration on the loop so the additive
// tree can still be rebuilt after the lane stages have stood down. Seedless,
// so the initial value joins the leaves and the depth drops to its logarithm.
// CHECK-LABEL: func.func @rebuilds_a_stamped_fast_window(
// CHECK-NOT: scf.for
// CHECK: %[[A:.*]] = arith.addf %[[L0:.*]], %[[L1:.*]] : f32
// CHECK: %[[B:.*]] = arith.addf %[[L2:.*]], %[[L3:.*]] : f32
// CHECK: arith.addf %[[A]], %[[B]] {ondsp.fast_used = ["rebuild_reduction_tree"]} : f32
func.func @rebuilds_a_stamped_fast_window(%x: memref<8xf32>) -> f32 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %seed = memref.load %x[%c0] : memref<8xf32>
  %sum = scf.for %i = %c1 to %c4 step %c1 iter_args(%acc = %seed) -> (f32) {
    %v = memref.load %x[%i] : memref<8xf32>
    %n = arith.addf %acc, %v : f32
    scf.yield %n : f32
  } {ondsp.numeric = #ondsp.fp<format = f32, contract = fast>}
  return %sum : f32
}

// -----

// The same loop without the stamp is the declared left fold; the rebuild is a
// permission the declaration carries, not a shape this pass assumes.
// CHECK-LABEL: func.func @keeps_an_unstamped_window_ordered(
// CHECK: %[[A:.*]] = arith.addf %[[SEED:.*]], %{{.*}} : f32
// CHECK: %[[B:.*]] = arith.addf %[[A]], %{{.*}} : f32
// CHECK: arith.addf %[[B]], %{{.*}} : f32
// CHECK-NOT: ondsp.fast_used
func.func @keeps_an_unstamped_window_ordered(%x: memref<8xf32>) -> f32 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %seed = memref.load %x[%c0] : memref<8xf32>
  %sum = scf.for %i = %c1 to %c4 step %c1 iter_args(%acc = %seed) -> (f32) {
    %v = memref.load %x[%i] : memref<8xf32>
    %n = arith.addf %acc, %v : f32
    scf.yield %n : f32
  }
  return %sum : f32
}
