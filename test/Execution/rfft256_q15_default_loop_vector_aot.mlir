// RUN: ondrix-opt %S/rfft256_q15_aot.mlir --ondrix-default-pipeline="vector-bits=256 fft-loops=true" > %t.mlir
// RUN: FileCheck %s --input-file=%t.mlir --check-prefix=VEC --implicit-check-not="vector<8xi33>"
// RUN: ondrix-translate %t.mlir --mlir-to-llvmir > %t.ll
// RUN: opt -O3 -vectorize-loops=false -vectorize-slp=false %t.ll -o %t.bc
// RUN: llc -O2 -mtriple=x86_64-unknown-linux-gnu -mattr=+avx2 -relocation-model=pic -filetype=obj %t.bc -o %t.o
// RUN: cc -DTEST_CANONICAL_OUT_PARAMS %S/Inputs/rfft256_q15_aot.c %t.o -o %t
// RUN: %t

// The same declared width reaches the packed CFFT inside a real-spectrum
// producer, then matches its independent recursive reference.
// VEC: llvm.load {{.*}} : !llvm.ptr -> vector<16xi32>

module {}
