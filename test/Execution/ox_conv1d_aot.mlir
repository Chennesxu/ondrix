// RUN: ondrix-compile %S/../Frontend/Inputs/q15_convolution.ox > %t.q15.source.mlir
// RUN: ondrix-opt %t.q15.source.mlir --convert-ondrix-to-ondsp --convert-ondsp-fixed-to-scalar --empty-tensor-to-alloc-tensor --one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map allow-return-allocs" --expand-strided-metadata --lower-affine --convert-scf-to-cf --finalize-memref-to-llvm --convert-math-to-llvm --convert-arith-to-llvm --convert-cf-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts > %t.q15.mlir
// RUN: ondrix-translate %t.q15.mlir --mlir-to-llvmir > %t.q15.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.q15.ll -o %t.q15.o
// RUN: ondrix-compile %S/../Frontend/Inputs/f32_correlation.ox > %t.f32.source.mlir
// RUN: ondrix-opt %t.f32.source.mlir --convert-ondrix-to-ondsp --empty-tensor-to-alloc-tensor --one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map allow-return-allocs" --expand-strided-metadata --lower-affine --convert-scf-to-cf --finalize-memref-to-llvm --convert-math-to-llvm --convert-arith-to-llvm --convert-cf-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts > %t.f32.mlir
// RUN: ondrix-translate %t.f32.mlir --mlir-to-llvmir > %t.f32.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.f32.ll -o %t.f32.o
// RUN: cc %S/Inputs/ox_conv1d_aot.c %t.q15.o %t.f32.o -lm -o %t
// RUN: %t
