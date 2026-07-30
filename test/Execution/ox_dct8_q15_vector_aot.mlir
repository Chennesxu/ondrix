// RUN: ondrix-compile %S/../Frontend/Inputs/q15_dct8.ox > %t.source.mlir
// RUN: ondrix-opt %t.source.mlir --empty-tensor-to-alloc-tensor --one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map allow-return-allocs" --cse --canonicalize > %t.bufferized.mlir
// RUN: ondrix-opt %t.bufferized.mlir --vectorize-ondsp-constant-saturating-memref-reduce="vector-width=8 max-elements=64 proof-trace-output=%t.proof.json" > %t.proven.mlir
// RUN: FileCheck %s --check-prefix=PROVEN --implicit-check-not=ondsp.reduce_mac < %t.proven.mlir
// RUN: ondrix-opt %t.bufferized.mlir --verify-ondsp-constant-reassociation-proof-trace="proof-trace-input=%t.proof.json max-elements=64" > /dev/null
// RUN: ondrix-opt %t.proven.mlir --normalize-ondsp-fixed-vector-reduce --convert-ondsp-fixed-to-scalar --expand-strided-metadata --lower-affine --convert-scf-to-cf --convert-vector-to-llvm --finalize-memref-to-llvm --convert-math-to-llvm --convert-arith-to-llvm --convert-cf-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts > %t.mlir
// RUN: FileCheck %s --check-prefix=VECTOR < %t.mlir
// RUN: ondrix-translate %t.mlir --mlir-to-llvmir > %t.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.ll -o %t.o
// RUN: cc %S/Inputs/ox_dct8_q15_vector_aot.c %t.o -o %t
// RUN: %t
// RUN: llc -O2 -mtriple=x86_64-unknown-linux-gnu -mattr=+avx2 -filetype=asm %t.ll -o %t.s
// RUN: FileCheck %s --check-prefix=AVX2 < %t.s

// End-to-end chain witness for the .ox surface through the OTHER legality
// route: the same pinned frontend fixture that test/Frontend/dct.mlir checks
// compiles through the constant-coefficient prefix-range-proof Vector route
// down to an executed object, with the emitted proof trace replayed against
// the original bufferized IR. N = 8 is exactly one width-8 chunk per row, so
// every row reduction must be proof-authorized: a memref-form
// ondsp.reduce_mac surviving the vectorize pass means a row was refused, and
// the implicit check fails — a silent ordered fallback cannot pass this
// witness. Schedule coverage (chained chunks at N = 32, directed corner
// rails) stays with dct_q15_vector_aot.mlir.

// PROVEN-LABEL: func.func @q15_dct8
// PROVEN: vector.reduction <add>, {{.*}} : vector<8xi64> into i64

// No ondsp operation may survive the pipeline.
// VECTOR-NOT: ondsp.

// The label is anchored to line start so it cannot match the
// `_mlir_ciface_...` interface thunk emitted for the same kernel.
// AVX2-LABEL: {{^}}q15_dct8:
// AVX2: vpmulld
