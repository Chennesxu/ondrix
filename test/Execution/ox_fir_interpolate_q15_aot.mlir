// RUN: ondrix-compile %S/../Frontend/Inputs/q15_fir_interpolate.ox > %t.source.mlir
// RUN: FileCheck %s --check-prefix=SOURCE < %t.source.mlir
// RUN: ondrix-opt %t.source.mlir --convert-ondrix-to-ondsp --convert-ondsp-fixed-to-scalar --empty-tensor-to-alloc-tensor --one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map allow-return-allocs" --expand-strided-metadata --lower-affine --convert-scf-to-cf --finalize-memref-to-llvm --convert-arith-to-llvm --convert-cf-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts > %t.mlir
// RUN: ondrix-translate %t.mlir --mlir-to-llvmir > %t.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.ll -o %t.o
// RUN: cc -DFIR_INTERPOLATE_SYMBOL=_mlir_ciface_q15_fir_interpolate -DFIR_INTERPOLATE_ACCUMULATOR_WIDTH=33 -DFIR_INTERPOLATE_UPDATE_OVERFLOW=WRAP -DFIR_INTERPOLATE_OUTPUT_ROUNDING=NEAREST_TIES_POSITIVE %S/Inputs/fir_interpolate_q15_aot.c %t.o -o %t
// RUN: %t

// SOURCE-LABEL: func.func @q15_fir_interpolate
// SOURCE: ondrix.fir_interpolate
// SOURCE-SAME: accumulator = !ondsp.acc<storage = i33, frac = 30, signed, update_overflow = wrap>
// SOURCE-SAME: factor = 2
// SOURCE-SAME: overflow = #ondsp.overflow<saturate>
// SOURCE-SAME: rounding = #ondsp.rounding<nearest_ties_positive>
