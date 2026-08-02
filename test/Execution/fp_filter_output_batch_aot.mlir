// RUN: ondrix-opt %s --empty-tensor-to-alloc-tensor --one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map" --cse --canonicalize --vectorize-ondsp-fp-filter-outputs="vector-width=8" > %t.batched.mlir
// RUN: FileCheck %s --check-prefix=BATCHED < %t.batched.mlir
// RUN: ondrix-opt %t.batched.mlir --lower-ondsp-f32-reduce-to-scalar --convert-ondsp-fixed-to-scalar --expand-strided-metadata --lower-affine --convert-scf-to-cf --convert-vector-to-llvm --finalize-memref-to-llvm --convert-math-to-llvm --convert-arith-to-llvm --convert-cf-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts > %t.mlir
// RUN: FileCheck %s --check-prefix=LOWERED --implicit-check-not=ondsp. < %t.mlir
// RUN: ondrix-translate %t.mlir --mlir-to-llvmir > %t.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.ll -o %t.o
// RUN: cc -ffp-contract=off %S/Inputs/fp_filter_output_batch_aot.c %t.o -lm -o %t
// RUN: %t
// RUN: llc -O2 -mtriple=x86_64-unknown-linux-gnu -mattr=+avx2,+fma -filetype=asm %t.ll -o %t.s
// RUN: FileCheck %s --check-prefix=AVX2 < %t.s

// Vertical output batching for the valid-boundary f32 filter, as object
// evidence. This is the second incarnation of the order-preserving axis: the
// fixed-point decimate batching established it, and this one carries it into
// floating point, where the preserved thing is not an accumulator rail but a
// per-lane event graph.
//
// The lanes of the batched accumulator carry INDEPENDENT outputs. Lane j is
// output m + j, visits the same taps in the same increasing order, and applies
// the same declared update — one rounded product then one accumulation under
// contract=off, one fused event under contract=fma, both from +0.0 — to its
// own accumulator. Nothing is reassociated and no lane is ever combined with
// another, so this rewrite needs no reassociation proof, unlike the horizontal
// f32 reductions whose lanes do reorder the fold.
//
// off and fma are both EXACT contracts: they name the event graph rather than
// permitting a relation, so the gate is bit-for-bit rather than a tolerance.
// That is what makes the two profiles worth running side by side — a batched
// body that fused an off tap, or split an fma tap, changes exported bits on
// the directed corpus and cannot hide inside an error bound.
//
// Four kernels share one object. The two static kernels are batched; the two
// dynamic ones keep the ordered schedule because the pass refuses a loop whose
// output length and input extent are not statically known. That refusal is
// what gives the harness an in-object ordered oracle rather than a
// recompilation: the same object holds both schedules for both contracts.
//
// The output length 33 is not a multiple of the width, so with width 8 the
// batched loop covers outputs 0..31 and the untouched ordered loop covers
// output 32. Both paths run in every corpus entry.

// BATCHED-LABEL: func.func @f32_filter_off_batched
// BATCHED: %[[OFF_END:.*]] = arith.constant 32 : index
// BATCHED: %[[OFF_STEP:.*]] = arith.constant 8 : index
// BATCHED: scf.for %{{.*}} = %{{.*}} to %[[OFF_END]] step %[[OFF_STEP]] {
// The off lanes keep the two declared events per tap, and neither becomes a
// fused operation.
// BATCHED: vector.load {{.*}} : memref<40xf32>, vector<8xf32>
// BATCHED: arith.mulf {{.*}} : vector<8xf32>
// BATCHED: arith.addf {{.*}} : vector<8xf32>
// BATCHED-NOT: math.fma
// BATCHED: vector.store {{.*}} : memref<33xf32>, vector<8xf32>
// The remainder keeps the ordered reduction untouched.
// BATCHED: scf.for %{{.*}} = %[[OFF_END]] to %{{.*}} step %{{.*}} {
// BATCHED: ondsp.reduce_mac

// BATCHED-LABEL: func.func @f32_filter_fma_batched
// BATCHED: %[[FMA_END:.*]] = arith.constant 32 : index
// BATCHED: %[[FMA_STEP:.*]] = arith.constant 8 : index
// BATCHED: scf.for %{{.*}} = %{{.*}} to %[[FMA_END]] step %[[FMA_STEP]] {
// The fma lanes keep one fused event per tap, with no separate product.
// BATCHED: vector.load {{.*}} : memref<40xf32>, vector<8xf32>
// BATCHED: math.fma {{.*}} : vector<8xf32>
// BATCHED-NOT: arith.mulf {{.*}} : vector<8xf32>
// BATCHED: vector.store {{.*}} : memref<33xf32>, vector<8xf32>
// BATCHED: scf.for %{{.*}} = %[[FMA_END]] to %{{.*}} step %{{.*}} {
// BATCHED: ondsp.reduce_mac

// BATCHED-LABEL: func.func @f32_filter_off_ordered
// BATCHED-NOT: vector.load
// BATCHED: ondsp.reduce_mac

// BATCHED-LABEL: func.func @f32_filter_fma_ordered
// BATCHED-NOT: vector.load
// BATCHED: ondsp.reduce_mac

// LOWERED-LABEL: llvm.func @f32_filter_off_batched
// LOWERED-LABEL: llvm.func @f32_filter_fma_batched
// LOWERED-LABEL: llvm.func @f32_filter_off_ordered
// LOWERED-LABEL: llvm.func @f32_filter_fma_ordered

// The declared contract must survive all the way into selected instructions:
// a packed fused multiply-add for fma, and a packed multiply followed by a
// packed add — never a fused form — for off. Anything else would mean the
// batched body reached the backend as a relation rather than as the event
// graph the contract names.
// AVX2-LABEL: {{^}}f32_filter_off_batched:
// AVX2: vmulps
// AVX2: vaddps
// AVX2-NOT: vfmadd
// AVX2-LABEL: {{^}}f32_filter_fma_batched:
// AVX2: vfmadd{{[0-9]+}}ps

func.func @f32_filter_off_batched(
    %input: tensor<40xf32>, %coeffs: tensor<8xf32>, %init: tensor<33xf32>)
    -> tensor<33xf32> attributes {llvm.emit_c_interface} {
  %result = ondrix.fir_filter %input, %coeffs, %init {
    boundary = #ondrix.fir_boundary<valid>,
    numeric = #ondsp.fp<format = f32, contract = off>
  } : (tensor<40xf32>, tensor<8xf32>, tensor<33xf32>) -> tensor<33xf32>
  return %result : tensor<33xf32>
}

func.func @f32_filter_fma_batched(
    %input: tensor<40xf32>, %coeffs: tensor<8xf32>, %init: tensor<33xf32>)
    -> tensor<33xf32> attributes {llvm.emit_c_interface} {
  %result = ondrix.fir_filter %input, %coeffs, %init {
    boundary = #ondrix.fir_boundary<valid>,
    numeric = #ondsp.fp<format = f32, contract = fma>
  } : (tensor<40xf32>, tensor<8xf32>, tensor<33xf32>) -> tensor<33xf32>
  return %result : tensor<33xf32>
}

// The dynamic signatures are the in-object ordered oracles: the pass cannot
// see the output length or the input extent, so it leaves these loops exactly
// as the bufferization emitted them. The harness calls them with the same
// extents as the static kernels, so the two schedules meet on one corpus.
func.func @f32_filter_off_ordered(
    %input: tensor<?xf32>, %coeffs: tensor<?xf32>, %init: tensor<?xf32>)
    -> tensor<?xf32> attributes {llvm.emit_c_interface} {
  %result = ondrix.fir_filter %input, %coeffs, %init {
    boundary = #ondrix.fir_boundary<valid>,
    numeric = #ondsp.fp<format = f32, contract = off>
  } : (tensor<?xf32>, tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  return %result : tensor<?xf32>
}

func.func @f32_filter_fma_ordered(
    %input: tensor<?xf32>, %coeffs: tensor<?xf32>, %init: tensor<?xf32>)
    -> tensor<?xf32> attributes {llvm.emit_c_interface} {
  %result = ondrix.fir_filter %input, %coeffs, %init {
    boundary = #ondrix.fir_boundary<valid>,
    numeric = #ondsp.fp<format = f32, contract = fma>
  } : (tensor<?xf32>, tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  return %result : tensor<?xf32>
}
