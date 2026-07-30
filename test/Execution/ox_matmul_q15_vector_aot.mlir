// RUN: ondrix-compile %S/../Frontend/Inputs/q15_matmul.ox > %t.source.mlir
// RUN: ondrix-opt %t.source.mlir --empty-tensor-to-alloc-tensor --one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map allow-return-allocs" --cse --canonicalize > %t.bufferized.mlir
// RUN: ondrix-opt %t.bufferized.mlir --vectorize-ondsp-fixed-memref-reduce="vector-width=8" --parallelize-ondsp-fixed-wrap-vector-reduce > %t.parallel.mlir
// RUN: FileCheck %s --check-prefix=PARALLEL --implicit-check-not=ondsp.reduce_mac < %t.parallel.mlir
// RUN: ondrix-opt %t.parallel.mlir --normalize-ondsp-fixed-vector-reduce --convert-ondsp-fixed-to-scalar --expand-strided-metadata --lower-affine --convert-scf-to-cf --convert-vector-to-llvm --finalize-memref-to-llvm --convert-math-to-llvm --convert-arith-to-llvm --convert-cf-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts > %t.mlir
// RUN: FileCheck %s --check-prefix=VECTOR < %t.mlir
// RUN: ondrix-translate %t.mlir --mlir-to-llvmir > %t.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.ll -o %t.o
// RUN: cc %S/Inputs/ox_matmul_q15_vector_aot.c %t.o -o %t
// RUN: %t
// RUN: llc -O2 -mtriple=x86_64-unknown-linux-gnu -mattr=+avx2 -filetype=asm %t.ll -o %t.s
// RUN: FileCheck %s --check-prefix=AVX2 < %t.s

// End-to-end chain witness for the .ox surface: the same pinned frontend
// fixture that test/Frontend/matmul.mlir checks compiles through the
// exact-modulo wrap Vector route down to an executed object. The pinned
// 4x8 * 8x3 shape has K = 8, exactly one width-8 chunk per output element,
// so if the wrap legality gate refused the frontend-emitted form the
// memref-form ondsp.reduce_mac would survive and the implicit check below
// would fail: a silent scalar fallback cannot pass this witness. Schedule
// coverage (scalar tails, extreme extents, saturating corners) stays with
// matmul_q15_vector_aot.mlir; this test only pins that the frontend-emitted
// op composes with that path unchanged.

// PARALLEL-LABEL: func.func @q15_matmul
// PARALLEL: vector.reduction <add>, {{.*}} : vector<8xi64> into i64
// PARALLEL: ondsp.acc_add_term

// No ondsp operation may survive the pipeline.
// VECTOR-NOT: ondsp.

// The label is anchored to line start so it cannot match the
// `_mlir_ciface_...` interface thunk emitted for the same kernel.
// AVX2-LABEL: {{^}}q15_matmul:
// AVX2: vpmulld
