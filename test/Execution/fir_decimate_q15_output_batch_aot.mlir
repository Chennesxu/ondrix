// RUN: ondrix-opt %s --one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map" --cse --canonicalize --vectorize-ondsp-fixed-decimate-outputs="vector-width=8" > %t.batched.mlir
// RUN: FileCheck %s --check-prefix=BATCHED < %t.batched.mlir
// RUN: ondrix-opt %t.batched.mlir --convert-ondsp-fixed-to-scalar --expand-strided-metadata --lower-affine --convert-scf-to-cf --convert-vector-to-llvm --finalize-memref-to-llvm --convert-arith-to-llvm --convert-cf-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts > %t.mlir
// RUN: FileCheck %s --check-prefix=LOWERED --implicit-check-not=ondsp. < %t.mlir
// RUN: ondrix-translate %t.mlir --mlir-to-llvmir > %t.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.ll -o %t.o
// RUN: cc %S/Inputs/fir_decimate_q15_output_batch_aot.c %t.o -o %t
// RUN: %t
// RUN: llc -O2 -mtriple=x86_64-unknown-linux-gnu -mattr=+avx2 -filetype=asm %t.ll -o %t.s
// RUN: FileCheck %s --check-prefix=AVX2 < %t.s

// Vertical output batching for Q15 decimation, as object evidence.
//
// The lanes of the batched accumulator carry INDEPENDENT outputs. Lane j is
// output m + j, visits the same taps in the same increasing order, and applies
// the same declared update to its own accumulator. Nothing is reassociated, so
// this rewrite needs no range or overflow proof at all — the contrast with the
// horizontal reduction passes, whose lanes do reorder the fold and therefore
// do require one, is the point of committing both.
//
// The gate compiles four kernels into one object. The static pair is batched;
// the dynamic pair keeps the ordered schedule because the pass refuses a loop
// whose output length and input extent are not statically known, which gives
// the harness an in-object ordered oracle rather than a recompilation. The i34
// profile is the discriminating accumulator: eight products of
// -32768 * -32768 = 2^30 sum to exactly 2^33, one past the i34 maximum, so a
// saturating i34 accumulator clamps on its last tap and the directed rail
// corpus exercises that clamp per lane inside the batched body. The i40
// profile carries the same corpus without reaching its own rail.
//
// The output length 19 is odd, so with width 8 the batched loop covers outputs
// 0..15 and the untouched ordered loop covers 16..18. Both paths run in every
// trial.

// BATCHED-LABEL: func.func @decimate_i40_batched
// BATCHED: %[[BATCHED_END:.*]] = arith.constant 16 : index
// BATCHED: %[[BATCH_STEP:.*]] = arith.constant 8 : index
// BATCHED: scf.for %[[BLOCK:.*]] = %{{.*}} to %[[BATCHED_END]] step %[[BATCH_STEP]] {
// BATCHED: ondsp.acc_zero : <storage = i40, frac = 30, signed, update_overflow = saturate, lanes = 8>
// The batched block reads one contiguous factor*width span per tap and keeps
// the phase-zero lanes.
// BATCHED-COUNT-8: vector.load {{.*}} : memref<44xi16>, vector<16xi16>
// BATCHED: ondsp.acc_export
// BATCHED-SAME: update_overflow = saturate, lanes = 8>) -> vector<8xi16>
// BATCHED: vector.store {{.*}} : memref<19xi16>, vector<8xi16>
// The remainder keeps the ordered reduction untouched.
// BATCHED: scf.for %{{.*}} = %[[BATCHED_END]] to %{{.*}} step %{{.*}} {
// BATCHED: ondsp.reduce_mac

// BATCHED-LABEL: func.func @decimate_i40_ordered
// BATCHED-NOT: vector.load
// BATCHED: ondsp.reduce_mac

// BATCHED-LABEL: func.func @decimate_i34_batched
// BATCHED: ondsp.mac
// BATCHED-SAME: update_overflow = saturate, lanes = 8>, vector<8xi16>, i16)

// BATCHED-LABEL: func.func @decimate_i34_ordered
// BATCHED-NOT: vector.load
// BATCHED: ondsp.reduce_mac

// LOWERED-LABEL: llvm.func @decimate_i40_batched
// LOWERED-LABEL: llvm.func @decimate_i34_batched

// The batched body must reach real vector multiplies, and the multi-lane
// accumulator must stay in vector registers: a lane-by-lane extraction storm
// would mean the lanes were spilled back to scalars and the batching bought
// nothing. Both extraction forms are pinned absent in the whole object.
// AVX2: vpmul{{lw|ld}}
// AVX2-NOT: vpextrw
// AVX2-NOT: vpextrq

func.func @decimate_i40_batched(
    %input: tensor<44xi16>, %coeffs: tensor<8xi16>, %init: tensor<19xi16>)
    -> tensor<19xi16> attributes {llvm.emit_c_interface} {
  %result = ondrix.fir_decimate %input, %coeffs, %init {
    accumulator = !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>,
    dst = #ondsp.fixed<signed, storage = i16, frac = 15>,
    factor = 2,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    overflow = #ondsp.overflow<saturate>,
    product = #ondsp.product<full>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<44xi16>, tensor<8xi16>, tensor<19xi16>) -> tensor<19xi16>
  return %result : tensor<19xi16>
}

// The dynamic signature is the in-object ordered oracle: the pass cannot know
// the output length or the input extent statically, so it leaves this loop
// exactly as the bufferization emitted it.
func.func @decimate_i40_ordered(
    %input: tensor<?xi16>, %coeffs: tensor<?xi16>, %init: tensor<?xi16>)
    -> tensor<?xi16> attributes {llvm.emit_c_interface} {
  %result = ondrix.fir_decimate %input, %coeffs, %init {
    accumulator = !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>,
    dst = #ondsp.fixed<signed, storage = i16, frac = 15>,
    factor = 2,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    overflow = #ondsp.overflow<saturate>,
    product = #ondsp.product<full>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<?xi16>, tensor<?xi16>, tensor<?xi16>) -> tensor<?xi16>
  return %result : tensor<?xi16>
}

// The i34 accumulator reaches its saturation rail exactly at eight taps, so
// the per-lane clamp is observable in the batched body.
func.func @decimate_i34_batched(
    %input: tensor<44xi16>, %coeffs: tensor<8xi16>, %init: tensor<19xi16>)
    -> tensor<19xi16> attributes {llvm.emit_c_interface} {
  %result = ondrix.fir_decimate %input, %coeffs, %init {
    accumulator = !ondsp.acc<storage = i34, frac = 30, signed, update_overflow = saturate>,
    dst = #ondsp.fixed<signed, storage = i16, frac = 15>,
    factor = 2,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    overflow = #ondsp.overflow<saturate>,
    product = #ondsp.product<full>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<44xi16>, tensor<8xi16>, tensor<19xi16>) -> tensor<19xi16>
  return %result : tensor<19xi16>
}

func.func @decimate_i34_ordered(
    %input: tensor<?xi16>, %coeffs: tensor<?xi16>, %init: tensor<?xi16>)
    -> tensor<?xi16> attributes {llvm.emit_c_interface} {
  %result = ondrix.fir_decimate %input, %coeffs, %init {
    accumulator = !ondsp.acc<storage = i34, frac = 30, signed, update_overflow = saturate>,
    dst = #ondsp.fixed<signed, storage = i16, frac = 15>,
    factor = 2,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    overflow = #ondsp.overflow<saturate>,
    product = #ondsp.product<full>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<?xi16>, tensor<?xi16>, tensor<?xi16>) -> tensor<?xi16>
  return %result : tensor<?xi16>
}
