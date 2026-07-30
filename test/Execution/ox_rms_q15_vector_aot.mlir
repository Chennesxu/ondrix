// RUN: ondrix-compile %S/../Frontend/Inputs/q15_rms.ox > %t.nearest.source.mlir
// RUN: ondrix-opt %t.nearest.source.mlir --empty-tensor-to-alloc-tensor --one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map allow-return-allocs" --cse --canonicalize --vectorize-ondsp-fixed-memref-reduce="vector-width=8" --parallelize-ondsp-fixed-wrap-vector-reduce > %t.nearest.parallel.mlir
// RUN: FileCheck %s --check-prefix=NEAREST --implicit-check-not=ondsp.reduce_mac < %t.nearest.parallel.mlir
// RUN: ondrix-opt %t.nearest.parallel.mlir --normalize-ondsp-fixed-vector-reduce --convert-ondsp-fixed-to-scalar --expand-strided-metadata --lower-affine --convert-scf-to-cf --convert-vector-to-llvm --finalize-memref-to-llvm --convert-math-to-llvm --convert-arith-to-llvm --convert-cf-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts > %t.nearest.mlir
// RUN: FileCheck %s --check-prefix=VECTOR < %t.nearest.mlir
// RUN: ondrix-translate %t.nearest.mlir --mlir-to-llvmir > %t.nearest.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.nearest.ll -o %t.nearest.o

// RUN: ondrix-compile %S/../Frontend/Inputs/q15_rms_floor.ox > %t.floor.source.mlir
// RUN: ondrix-opt %t.floor.source.mlir --empty-tensor-to-alloc-tensor --one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map allow-return-allocs" --cse --canonicalize --vectorize-ondsp-fixed-memref-reduce="vector-width=8" --parallelize-ondsp-fixed-wrap-vector-reduce > %t.floor.parallel.mlir
// RUN: FileCheck %s --check-prefix=FLOOR --implicit-check-not=ondsp.reduce_mac < %t.floor.parallel.mlir
// RUN: ondrix-opt %t.floor.parallel.mlir --normalize-ondsp-fixed-vector-reduce --convert-ondsp-fixed-to-scalar --expand-strided-metadata --lower-affine --convert-scf-to-cf --convert-vector-to-llvm --finalize-memref-to-llvm --convert-math-to-llvm --convert-arith-to-llvm --convert-cf-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts > %t.floor.mlir
// RUN: FileCheck %s --check-prefix=VECTOR < %t.floor.mlir
// RUN: ondrix-translate %t.floor.mlir --mlir-to-llvmir > %t.floor.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.floor.ll -o %t.floor.o

// RUN: cc %S/Inputs/ox_rms_q15_vector_aot.c %t.nearest.o %t.floor.o -o %t -lm
// RUN: %t

// End-to-end chain witness for the .ox surface: the two pinned frontend
// fixtures that test/Frontend/rms.mlir checks — the nearest_even default and
// the per-call-site `root_rounding=toward_negative` — compile through the
// exact-modulo wrap Vector route down to one executed object each. N = 64 at
// width 8 leaves no scalar tail, so a memref-form ondsp.reduce_mac surviving
// either pipeline means the wrap legality gate refused the frontend-emitted
// form, and the implicit check fails: a silent scalar fallback cannot pass
// this witness. The harness drives a directed input where the two declared
// root roundings must disagree, so the source-level rounding parameter is
// checked all the way to executed bits, not only to the IR attribute.
// Schedule coverage (scalar tails, the 4096-point extent, saturating rails)
// stays with rms_q15_vector_aot.mlir.

// NEAREST-LABEL: func.func @q15_rms
// NEAREST: vector.reduction <add>, {{.*}} : vector<8xi64> into i64
// NEAREST: ondsp.acc_add_term
// NEAREST: ondsp.sqrt_fixed {{.*}}rounding = #ondsp.rounding<nearest_even>

// FLOOR-LABEL: func.func @q15_rms_floor
// FLOOR: vector.reduction <add>, {{.*}} : vector<8xi64> into i64
// FLOOR: ondsp.acc_add_term
// FLOOR: ondsp.sqrt_fixed {{.*}}rounding = #ondsp.rounding<toward_negative>

// No ondsp operation may survive either pipeline.
// VECTOR-NOT: ondsp.
