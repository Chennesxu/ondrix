// RUN: ondrix-compile %S/../Frontend/Inputs/q15_fir_auto.ox > %t.source.mlir
// RUN: FileCheck %s --check-prefix=SOURCE < %t.source.mlir
// RUN: ondrix-opt %t.source.mlir --convert-ondrix-to-ondsp --convert-ondsp-fixed-to-scalar --convert-scf-to-cf --finalize-memref-to-llvm --convert-arith-to-llvm --convert-cf-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts > %t.mlir
// RUN: FileCheck %s --check-prefix=LOWERED < %t.mlir
// RUN: ondrix-translate %t.mlir --mlir-to-llvmir > %t.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.ll -o %t.o
// RUN: cc -DOX_AUTO_PROFILE %S/Inputs/ox_q15_fir_aot.c %t.o -o %t
// RUN: %t

// SOURCE-LABEL: func.func @q15_fir_auto
// SOURCE: !ondsp.acc<storage = i34, frac = 30, signed, update_overflow = wrap>

// LOWERED-LABEL: llvm.func @q15_fir_auto
// LOWERED: llvm.mul
// LOWERED-NOT: ondrix.
// LOWERED-NOT: ondsp.
