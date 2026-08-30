// RUN: ondrix-opt %s --convert-ondrix-to-ondsp --empty-tensor-to-alloc-tensor --one-shot-bufferize="bufferize-function-boundaries allow-return-allocs function-boundary-type-conversion=identity-layout-map" --cse --canonicalize --vectorize-ondsp-fixed-elementwise-updates="vector-width=8" > %t.batched.mlir
// RUN: FileCheck %s --check-prefix=BATCHED < %t.batched.mlir
// RUN: ondrix-opt %t.batched.mlir --convert-ondsp-fixed-to-scalar --expand-strided-metadata --lower-affine --convert-scf-to-cf --convert-vector-to-llvm --finalize-memref-to-llvm --convert-arith-to-llvm --convert-cf-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts > %t.mlir
// RUN: FileCheck %s --check-prefix=LOWERED --implicit-check-not=ondsp. < %t.mlir
// RUN: ondrix-translate %t.mlir --mlir-to-llvmir > %t.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.ll -o %t.o
// RUN: cc %S/Inputs/lms_q15_update_batch_aot.c %t.o -o %t
// RUN: %t

// Eleven taps at width eight: the batched block covers 0..7 and 8..10 stay on
// the untouched ordered loop, so both paths run in every trial.

// BATCHED-LABEL: func.func @lms_q15
// The steady-state update loads the span forward and reverses the lanes onto
// the backward sample walk; the harness is what proves that order.
// BATCHED: vector.shuffle %{{.*}} [7, 6, 5, 4, 3, 2, 1, 0] : vector<8xi16>, vector<8xi16>
// BATCHED: ondsp.round_shift %{{.*}} : (vector<8xi32>) -> vector<8xi16>
// BATCHED: ondsp.sat_cast %{{.*}} : (vector<8xi32>) -> vector<8xi16>
// BATCHED: vector.store %{{.*}} : memref<11xi16>, vector<8xi16>
// BATCHED: scf.for
// BATCHED: ondsp.sat_cast %{{.*}} : (i32) -> i16
// BATCHED: memref.store %{{.*}}, %{{.*}} : memref<11xi16>

// LOWERED-LABEL: llvm.func @lms_q15

func.func @lms_q15(%x: tensor<64xi16>, %d: tensor<64xi16>, %w: tensor<11xi16>)
    -> (tensor<64xi16>, tensor<11xi16>) attributes {llvm.emit_c_interface} {
  %error, %adapted = ondrix.lms %x, %d, %w {
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    rounding = #ondsp.rounding<nearest_even>,
    step_size = 4096 : i64
  } : (tensor<64xi16>, tensor<64xi16>, tensor<11xi16>) -> (tensor<64xi16>, tensor<11xi16>)
  return %error, %adapted : tensor<64xi16>, tensor<11xi16>
}
