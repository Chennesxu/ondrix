// RUN: ondrix-compile %S/../Frontend/Inputs/q15_cic_decimate.ox > %t.source.mlir
// RUN: FileCheck %s --check-prefix=SOURCE < %t.source.mlir
// RUN: ondrix-opt %t.source.mlir --convert-ondrix-to-ondsp --convert-ondsp-fixed-to-scalar --empty-tensor-to-alloc-tensor --one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map allow-return-allocs" --expand-strided-metadata --lower-affine --convert-scf-to-cf --finalize-memref-to-llvm --convert-arith-to-llvm --convert-cf-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts > %t.mlir
// RUN: ondrix-translate %t.mlir --mlir-to-llvmir > %t.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.ll -o %t.o
// RUN: cc -DCIC_SOURCE_SYMBOL=_mlir_ciface_q15_cic_decimate %S/Inputs/cic_decimate_q15_aot.c %t.o -o %t
// RUN: %t

// The source declaration must reach the emitted contract, not just a
// plausible cic operation: the same reference and corpus that gate the
// hand-written profile run against the object this .ox program produces.

// SOURCE: ondrix.cic_decimate
// SOURCE-SAME: overflow = #ondsp.overflow<wrap>
// SOURCE-SAME: rate = 4 : i64
// SOURCE-SAME: stages = 2 : i64
