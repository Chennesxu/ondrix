// RUN: ondrix-compile %S/../Frontend/Inputs/q15_butterfly.ox > %t.source.mlir
// RUN: ondrix-opt %t.source.mlir --convert-ondrix-to-ondsp --convert-ondsp-fixed-to-scalar --convert-arith-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts > %t.mlir
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ondrix-translate %t.mlir --mlir-to-llvmir > %t.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.ll -o %t.o
// RUN: cc -DOX_MULTI_RESULT %S/Inputs/cx_butterfly_q15_aot.c %t.o -o %t
// RUN: %t

// CHECK-NOT: ondrix.
// CHECK-NOT: ondsp.
// CHECK: llvm.func @_mlir_ciface_q15_butterfly
