// RUN: ondrix-compile %S/../Frontend/Inputs/q15_fir_stream.ox > %t.source.mlir
// RUN: FileCheck %s --check-prefix=SOURCE < %t.source.mlir
// RUN: ondrix-opt %t.source.mlir --convert-ondrix-to-ondsp --convert-ondsp-fixed-to-scalar --empty-tensor-to-alloc-tensor --one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map allow-return-allocs" --expand-strided-metadata --lower-affine --convert-scf-to-cf --finalize-memref-to-llvm --convert-math-to-llvm --convert-arith-to-llvm --convert-cf-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts > %t.mlir
// RUN: FileCheck %s --check-prefix=LOWERED < %t.mlir
// RUN: ondrix-translate %t.mlir --mlir-to-llvmir > %t.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.ll -o %t.o
// RUN: cc %S/Inputs/ox_q15_fir_stream_aot.c %t.o -o %t
// RUN: %t

// SOURCE-LABEL: func.func @q15_fir_stream
// SOURCE: ondrix.fir_stream
// SOURCE-SAME: !ondsp.acc<storage = i33, frac = 30, signed, update_overflow = wrap>

// LOWERED-LABEL: llvm.func @q15_fir_stream
// LOWERED-NOT: ondrix.
// LOWERED-NOT: ondsp.
// LOWERED-NOT: tensor.
