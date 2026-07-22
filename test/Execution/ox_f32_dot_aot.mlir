// RUN: ondrix-compile %S/../Frontend/Inputs/f32_dot_off.ox > %t.off.source.mlir
// RUN: ondrix-opt %t.off.source.mlir --convert-ondrix-to-ondsp --lower-ondsp-f32-reduce-to-scalar --convert-scf-to-cf --finalize-memref-to-llvm --convert-math-to-llvm --convert-arith-to-llvm --convert-cf-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts > %t.off.mlir
// RUN: ondrix-translate %t.off.mlir --mlir-to-llvmir > %t.off.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.off.ll -o %t.off.o
// RUN: ondrix-compile %S/../Frontend/Inputs/f32_dot_fma.ox > %t.fma.source.mlir
// RUN: ondrix-opt %t.fma.source.mlir --convert-ondrix-to-ondsp --lower-ondsp-f32-reduce-to-scalar --convert-scf-to-cf --finalize-memref-to-llvm --convert-math-to-llvm --convert-arith-to-llvm --convert-cf-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts > %t.fma.mlir
// RUN: ondrix-translate %t.fma.mlir --mlir-to-llvmir > %t.fma.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.fma.ll -o %t.fma.o
// RUN: cc %S/Inputs/ox_f32_dot_aot.c %t.off.o %t.fma.o -lm -o %t
// RUN: %t
