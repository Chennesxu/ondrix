// RUN: ondrix-compile %S/../Frontend/Inputs/q15_rfft_round_trip.ox > %t.source.mlir
// RUN: ondrix-opt %t.source.mlir --convert-ondrix-to-ondsp --convert-ondsp-fixed-to-scalar --empty-tensor-to-alloc-tensor --one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map allow-return-allocs" --expand-strided-metadata --lower-affine --convert-scf-to-cf --finalize-memref-to-llvm --convert-arith-to-llvm --convert-cf-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts > %t.mlir
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ondrix-translate %t.mlir --mlir-to-llvmir > %t.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.ll -o %t.o
// RUN: cc -DONDRIX_OX_RFFT_ROUND_TRIP %S/Inputs/rfft_q15_aot.c %t.o -o %t
// RUN: %t

// CHECK-NOT: ondrix.
// CHECK-NOT: ondsp.
// CHECK: llvm.func @_mlir_ciface_q15_rfft_round_trip
