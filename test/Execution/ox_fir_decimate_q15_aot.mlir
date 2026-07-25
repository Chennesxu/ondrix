// RUN: ondrix-compile %S/../Frontend/Inputs/q15_fir_decimate.ox > %t.source.mlir
// RUN: FileCheck %s --check-prefix=SOURCE < %t.source.mlir
// RUN: ondrix-opt %t.source.mlir --convert-ondrix-to-ondsp --convert-ondsp-fixed-to-scalar --empty-tensor-to-alloc-tensor --one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map allow-return-allocs" --expand-strided-metadata --lower-affine --convert-scf-to-cf --finalize-memref-to-llvm --convert-arith-to-llvm --convert-cf-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts > %t.mlir
// RUN: ondrix-translate %t.mlir --mlir-to-llvmir > %t.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.ll -o %t.o
// RUN: cc -DFIR_DECIMATE_SYMBOL=_mlir_ciface_q15_fir_decimate -DFIR_DECIMATE_ACCUMULATOR_WIDTH=34 -DFIR_DECIMATE_UPDATE_OVERFLOW=WRAP %S/Inputs/fir_decimate_q15_aot.c %t.o -o %t
// RUN: %t

// SOURCE-LABEL: func.func @q15_fir_decimate
// SOURCE: ondrix.fir_decimate
// SOURCE-SAME: accumulator = !ondsp.acc<storage = i34, frac = 30, signed, update_overflow = wrap>
// SOURCE-SAME: factor = 2
