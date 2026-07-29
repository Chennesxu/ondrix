// RUN: ondrix-opt %s --one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map allow-return-allocs" --cse --canonicalize --vectorize-ondsp-fixed-memref-reduce="vector-width=8" --parallelize-ondsp-fixed-wrap-vector-reduce --normalize-ondsp-fixed-vector-reduce --convert-ondsp-fixed-to-scalar --expand-strided-metadata --lower-affine --convert-scf-to-cf --convert-vector-to-llvm --finalize-memref-to-llvm --convert-arith-to-llvm --convert-cf-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts > %t.mlir
// RUN: FileCheck %s --check-prefix=VECTOR < %t.mlir
// RUN: ondrix-translate %t.mlir --mlir-to-llvmir > %t.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.ll -o %t.o
// RUN: cc %S/Inputs/rms_q15_vector_aot.c %t.o -o %t -lm
// RUN: %t
// RUN: llc -O2 -mtriple=x86_64-unknown-linux-gnu -mattr=+avx2 -filetype=asm %t.ll -o %t.s
// RUN: FileCheck %s --check-prefix=AVX2 < %t.s

// The bufferization consumer of ondrix.rms squares the input through one
// reduction whose two operands are the same buffer. Squares are at most 2^30
// and N is at most 4096, so the exact sum is at most 2^42: the i64 wrapping
// accumulator never wraps and is the exact-modulo reassociation class, which
// is what lets the horizontal Vector consumer fire without a prefix proof.
// The mean boundary exports at frac 30 - log2(N) so that acc_export divides
// by exactly the extent, and the declared root rounding routes to sqrt_fixed.
// The independent C reference is the same contract arithmetic that pins the
// tensor-form scalar gate, so this closes scalar == Vector == reference.

// No ondsp operation may survive the pipeline.
// VECTOR-NOT: ondsp.

// AVX2-LABEL: rms64_q15_vector:
// AVX2: vpmulld
// AVX2: vpaddq
// AVX2-LABEL: rms64_floor_q15_vector:
// AVX2: vpmulld
// AVX2: vpaddq
// AVX2-LABEL: rms4096_q15_vector:
// AVX2: vpmulld
// AVX2: vpaddq

func.func @rms64_q15_vector(%input: tensor<64xi16>) -> tensor<1xi16>
    attributes {llvm.emit_c_interface} {
  %result = ondrix.rms %input {
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<64xi16>) -> tensor<1xi16>
  return %result : tensor<1xi16>
}

func.func @rms64_floor_q15_vector(%input: tensor<64xi16>) -> tensor<1xi16>
    attributes {llvm.emit_c_interface} {
  %result = ondrix.rms %input {
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    rounding = #ondsp.rounding<toward_negative>
  } : (tensor<64xi16>) -> tensor<1xi16>
  return %result : tensor<1xi16>
}

// The largest admitted extent: 4096 squares of magnitude at most 2^30 sum to
// at most 2^42, and the mean boundary shifts by log2(4096) = 12.
func.func @rms4096_q15_vector(%input: tensor<4096xi16>) -> tensor<1xi16>
    attributes {llvm.emit_c_interface} {
  %result = ondrix.rms %input {
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<4096xi16>) -> tensor<1xi16>
  return %result : tensor<1xi16>
}
