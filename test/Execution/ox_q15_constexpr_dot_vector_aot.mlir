// RUN: ondrix-compile %S/../Frontend/Inputs/q15_dot_constexpr.ox > %t.source.mlir
// RUN: ondrix-opt %t.source.mlir --convert-ondrix-to-ondsp > %t.ondsp.mlir
// RUN: ondrix-opt %t.ondsp.mlir --vectorize-ondsp-constant-saturating-memref-reduce="vector-width=4 max-elements=64 proof-trace-output=%t.proof.json" > %t.proven.mlir
// RUN: FileCheck %s --check-prefix=PROVEN < %t.proven.mlir
// RUN: ondrix-opt %t.ondsp.mlir --verify-ondsp-constant-reassociation-proof-trace="proof-trace-input=%t.proof.json max-elements=64" > /dev/null
// RUN: ondrix-opt %t.proven.mlir --vectorize-ondsp-fixed-memref-reduce="vector-width=4" --parallelize-ondsp-fixed-wrap-vector-reduce --normalize-ondsp-fixed-vector-reduce --convert-ondsp-fixed-to-scalar --convert-scf-to-cf --convert-vector-to-llvm --finalize-memref-to-llvm --convert-arith-to-llvm --convert-cf-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts > %t.mlir
// RUN: ondrix-translate %t.mlir --mlir-to-llvmir > %t.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.ll -o %t.o
// RUN: cc %S/Inputs/ox_q15_constexpr_dot_vector_aot.c %t.o -o %t
// RUN: %t

// PROVEN-LABEL: func.func @q15_dot_constexpr
// PROVEN: vector.reduction <add>, {{.*}} : vector<4xi64> into i64
// PROVEN: ondsp.acc_add_term
// PROVEN-NOT: ondsp.reduce_mac
