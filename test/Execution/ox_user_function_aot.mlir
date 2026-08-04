// RUN: ondrix-compile %S/../Frontend/Inputs/q15_call_forward.ox > %t.call.mlir
// RUN: ondrix-opt %t.call.mlir --convert-ondrix-to-ondsp --convert-ondsp-fixed-to-scalar --empty-tensor-to-alloc-tensor --one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map allow-return-allocs" --expand-strided-metadata --lower-affine --convert-scf-to-cf --finalize-memref-to-llvm --convert-arith-to-llvm --convert-cf-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts > %t.call.llvm.mlir
// RUN: ondrix-translate %t.call.llvm.mlir --mlir-to-llvmir > %t.call.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.call.ll -o %t.call.o
// RUN: ondrix-compile %S/../Frontend/Inputs/q15_call_inlined.ox > %t.body.mlir
// RUN: ondrix-opt %t.body.mlir --convert-ondrix-to-ondsp --convert-ondsp-fixed-to-scalar --empty-tensor-to-alloc-tensor --one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map allow-return-allocs" --expand-strided-metadata --lower-affine --convert-scf-to-cf --finalize-memref-to-llvm --convert-arith-to-llvm --convert-cf-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts > %t.body.llvm.mlir
// RUN: ondrix-translate %t.body.llvm.mlir --mlir-to-llvmir > %t.body.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.body.ll -o %t.body.o
// RUN: cc %S/Inputs/ox_user_function_aot.c %t.call.o %t.body.o -o %t
// RUN: %t

// Calling a named body and writing that body at the call site must be the
// same program. Both objects run against one exact reference on a corpus
// whose products all reach the Q15 rail.
