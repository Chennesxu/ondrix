// RUN: ondrix-opt %S/cfft64_q15_aot.mlir --ondrix-default-pipeline="vector-bits=256 fft-loops=true" > %t.mlir
// RUN: FileCheck %s --input-file=%t.mlir --check-prefix=VEC --implicit-check-not="vector<8xi33>"
// RUN: ondrix-translate %t.mlir --mlir-to-llvmir > %t.ll
// RUN: llc -O2 -mtriple=x86_64-unknown-linux-gnu -mattr=+avx2 -relocation-model=pic -filetype=obj %t.ll -o %t.o
// RUN: cc -DTEST_CANONICAL_OUT_PARAMS %S/Inputs/cfft64_q15_aot.c %t.o -o %t
// RUN: %t

// The canonical loop schedule carries the declared AVX2 lane count into the
// already proven Q15 butterfly, then executes both directions against the
// independent staged reference and its directed rail corpus.
// VEC: llvm.load {{.*}} : !llvm.ptr -> vector<8xi32>

module {}
