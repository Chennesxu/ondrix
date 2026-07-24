// RUN: ondrix-compile %S/../Frontend/Inputs/q15_cfft8.ox > %t.source.mlir
// RUN: ondrix-opt %t.source.mlir --convert-ondrix-to-ondsp --convert-ondsp-fixed-to-scalar --empty-tensor-to-alloc-tensor --one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map allow-return-allocs" --expand-strided-metadata --lower-affine --convert-scf-to-cf --finalize-memref-to-llvm --convert-arith-to-llvm --convert-cf-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts > %t.mlir
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ondrix-translate %t.mlir --mlir-to-llvmir > %t.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.ll -o %t.o
// RUN: cc %S/Inputs/ox_cfft8_q15_aot.c %t.o -o %t
// RUN: %t
// RUN: ondrix-opt %t.source.mlir --convert-ondrix-to-ondsp="vectorize-static-cfft" --convert-ondsp-fixed-to-scalar --empty-tensor-to-alloc-tensor --one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map allow-return-allocs" --expand-strided-metadata --lower-affine --convert-scf-to-cf --convert-vector-to-llvm --finalize-memref-to-llvm --convert-arith-to-llvm --convert-cf-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts > %t.vector.mlir
// RUN: ondrix-translate %t.vector.mlir --mlir-to-llvmir > %t.vector.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.vector.ll -o %t.vector.o
// RUN: cc %S/Inputs/ox_cfft8_q15_aot.c %t.vector.o -o %t.vector
// RUN: %t.vector

// CHECK-NOT: ondrix.
// CHECK-NOT: ondsp.
// CHECK: llvm.func @_mlir_ciface_q15_cfft8
