// RUN: ondrix-opt %S/cfft_q31_aot.mlir --ondrix-default-pipeline="vector-bits=256 fft-loops=true" > %t.mlir
// RUN: FileCheck %s --input-file=%t.mlir --check-prefix=VEC --implicit-check-not="vector<8xi64>"
// RUN: ondrix-translate %t.mlir --mlir-to-llvmir > %t.ll
// RUN: llc -O2 -mtriple=x86_64-unknown-linux-gnu -mattr=+avx2 -relocation-model=pic -filetype=obj %t.ll -o %t.o
// RUN: cc -DTEST_CANONICAL_OUT_PARAMS %S/Inputs/cfft_q31_aot.c %t.o -o %t
// RUN: %t

// The packed-Q31 loop uses four i64 lanes in a 256-bit register. The
// independent reference checks forward, inverse and round-trip stages.
// VEC: llvm.load {{.*}} : !llvm.ptr -> vector<4xi64>

module {}
