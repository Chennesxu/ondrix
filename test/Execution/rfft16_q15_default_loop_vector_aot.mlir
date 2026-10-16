// RUN: ondrix-opt %S/rfft_q15_aot.mlir --ondrix-default-pipeline="vector-bits=256 fft-loops=true" > %t.mlir
// RUN: FileCheck %s --input-file=%t.mlir --check-prefix=VEC --implicit-check-not="vector<8xi33>"
// RUN: ondrix-translate %t.mlir --mlir-to-llvmir > %t.ll
// RUN: opt -O3 -vectorize-loops=false -vectorize-slp=false %t.ll -o %t.bc
// RUN: llc -O2 -mtriple=x86_64-unknown-linux-gnu -mattr=+avx2 -relocation-model=pic -filetype=obj %t.bc -o %t.o
// RUN: cc %S/Inputs/rfft_q15_aot.c %t.o -o %t
// RUN: %t

// RUN: ondrix-opt %S/rfft_q15_aot.mlir --ondrix-default-pipeline="vector-bits=128 fft-loops=true widening-multiply-low-halves=false multiply-add-adjacent-pairs=false" > %t.narrow.mlir
// RUN: FileCheck %s --input-file=%t.narrow.mlir --check-prefix=NARROW --implicit-check-not="vector<4xi33>"
// RUN: ondrix-translate %t.narrow.mlir --mlir-to-llvmir > %t.narrow.ll
// RUN: opt -O3 -vectorize-loops=false -vectorize-slp=false %t.narrow.ll -o %t.narrow.bc
// RUN: llc -O2 -mtriple=x86_64-unknown-linux-gnu -mattr=+avx2 -relocation-model=pic -filetype=obj %t.narrow.bc -o %t.narrow.o
// RUN: cc %S/Inputs/rfft_q15_aot.c %t.narrow.o -o %t.narrow
// RUN: %t.narrow

// The inverse and round-trip scalar-return producers use the same loop
// schedule and remain bit-exact under the independent Hermitian reference.
// VEC: llvm.load {{.*}} : !llvm.ptr -> vector<16xi32>

// NARROW: llvm.load {{.*}} : !llvm.ptr -> vector<8xi32>

module {}
