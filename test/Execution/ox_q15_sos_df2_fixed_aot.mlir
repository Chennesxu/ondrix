// RUN: ondrix-compile %S/../Frontend/Inputs/q15_sos_df2_fixed.ox > %t.source.mlir
// RUN: FileCheck %s --check-prefix=SOURCE < %t.source.mlir
// RUN: ondrix-opt %t.source.mlir --convert-ondrix-to-ondsp --convert-ondsp-fixed-to-scalar --empty-tensor-to-alloc-tensor --one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map allow-return-allocs" --expand-strided-metadata --lower-affine --convert-scf-to-cf --finalize-memref-to-llvm --convert-math-to-llvm --convert-arith-to-llvm --convert-cf-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts > %t.mlir
// RUN: FileCheck %s --check-prefix=LOWERED < %t.mlir
// RUN: ondrix-translate %t.mlir --mlir-to-llvmir > %t.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.ll -o %t.o
// RUN: cc %S/Inputs/ox_q15_sos_df2_fixed_aot.c %t.o -o %t
// RUN: %t

// SOURCE-LABEL: func.func @q15_sos_df2_fixed
// SOURCE: ondrix.sos_filter_df2_fixed
// SOURCE-SAME: !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>

// LOWERED-LABEL: llvm.func @q15_sos_df2_fixed
// LOWERED-NOT: ondrix.
// LOWERED-NOT: ondsp.
// LOWERED-NOT: tensor.
