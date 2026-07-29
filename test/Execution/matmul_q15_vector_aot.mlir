// RUN: ondrix-opt %s --one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map allow-return-allocs" --cse --canonicalize --vectorize-ondsp-fixed-memref-reduce="vector-width=8" --parallelize-ondsp-fixed-wrap-vector-reduce --normalize-ondsp-fixed-vector-reduce --convert-ondsp-fixed-to-scalar --expand-strided-metadata --lower-affine --convert-scf-to-cf --convert-vector-to-llvm --finalize-memref-to-llvm --convert-arith-to-llvm --convert-cf-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts > %t.mlir
// RUN: FileCheck %s --check-prefix=VECTOR < %t.mlir
// RUN: ondrix-translate %t.mlir --mlir-to-llvmir > %t.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.ll -o %t.o
// RUN: cc %S/Inputs/matmul_q15_vector_aot.c %t.o -o %t
// RUN: %t
// RUN: llc -O2 -mtriple=x86_64-unknown-linux-gnu -mattr=+avx2 -filetype=asm %t.ll -o %t.s
// RUN: FileCheck %s --check-prefix=AVX2 < %t.s

// The bufferization consumer of ondrix.matmul reaches the existing proof-gated
// Vector path: a transposed pack makes both reduction operands unit stride,
// the i40 wrapping accumulator is the exact-modulo reassociation class, and
// the horizontal sum therefore needs no prefix proof. The independent C
// reference is the same exact-i64-sum contract arithmetic that already pins
// the tensor-form scalar gate, so this closes scalar == Vector == reference.

// No ondsp operation may survive the pipeline.
// VECTOR-NOT: ondsp.

// AVX2-LABEL: matmul8x8x8_q15_vector:
// AVX2: vpmulld
// AVX2-LABEL: matmul4x16x3_q15_vector:
// AVX2: vpmulld
// AVX2-LABEL: matmul1x64x1_q15_vector:
// AVX2: vpmulld

func.func @matmul8x8x8_q15_vector(%a: tensor<8x8xi16>, %b: tensor<8x8xi16>) -> tensor<8x8xi16>
    attributes {llvm.emit_c_interface} {
  %c = ondrix.matmul %a, %b {
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<8x8xi16>, tensor<8x8xi16>) -> tensor<8x8xi16>
  return %c : tensor<8x8xi16>
}

func.func @matmul4x16x3_q15_vector(%a: tensor<4x16xi16>, %b: tensor<16x3xi16>) -> tensor<4x3xi16>
    attributes {llvm.emit_c_interface} {
  %c = ondrix.matmul %a, %b {
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<4x16xi16>, tensor<16x3xi16>) -> tensor<4x3xi16>
  return %c : tensor<4x3xi16>
}

// The longest admitted reduction: K = 64 products of magnitude at most 2^30
// sum to at most 2^36, the bound that keeps the i40 accumulator exact.
func.func @matmul1x64x1_q15_vector(%a: tensor<1x64xi16>, %b: tensor<64x1xi16>) -> tensor<1x1xi16>
    attributes {llvm.emit_c_interface} {
  %c = ondrix.matmul %a, %b {
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<1x64xi16>, tensor<64x1xi16>) -> tensor<1x1xi16>
  return %c : tensor<1x1xi16>
}
