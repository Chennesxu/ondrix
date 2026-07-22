// RUN: ondrix-compile %S/../Frontend/Inputs/q15_fir_constexpr.ox > %t.source.mlir
// RUN: ondrix-opt %t.source.mlir --specialize-ondrix-constant-fir --convert-ondrix-to-ondsp --convert-ondsp-fixed-to-scalar --convert-scf-to-cf --finalize-memref-to-llvm --convert-arith-to-llvm --convert-cf-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts > %t.mlir
// RUN: FileCheck %s --check-prefix=LOWERED < %t.mlir
// RUN: ondrix-translate %t.mlir --mlir-to-llvmir > %t.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.ll -o %t.o
// RUN: cc %S/Inputs/ox_q15_constexpr_fir_aot.c %t.o -o %t
// RUN: %t

// LOWERED-NOT: ondrix.
// LOWERED-NOT: ondsp.
// LOWERED: llvm.func @q15_fir_constexpr
