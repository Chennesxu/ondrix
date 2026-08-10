// RUN: ondrix-compile %S/../Frontend/Inputs/q15_fir_decimate.ox > %t.source.mlir
// RUN: FileCheck %s --check-prefix=SOURCE < %t.source.mlir
// RUN: ondrix-opt %t.source.mlir --convert-ondrix-to-ondsp --convert-ondsp-fixed-to-scalar --empty-tensor-to-alloc-tensor --one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map allow-return-allocs" --expand-strided-metadata --lower-affine --convert-scf-to-cf --finalize-memref-to-llvm --convert-arith-to-llvm --convert-cf-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts > %t.scalar.mlir
// RUN: ondrix-translate %t.scalar.mlir --mlir-to-llvmir > %t.scalar.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.scalar.ll -o %t.scalar.o
// RUN: cc -DFIR_DECIMATE_SYMBOL=_mlir_ciface_q15_fir_decimate -DFIR_DECIMATE_ACCUMULATOR_WIDTH=34 -DFIR_DECIMATE_UPDATE_OVERFLOW=WRAP -DFIR_DECIMATE_OUTPUT_ROUNDING=NEAREST_TIES_POSITIVE %S/Inputs/fir_decimate_q15_aot.c %t.scalar.o -o %t.scalar
// RUN: %t.scalar
// RUN: ondrix-opt %t.source.mlir --empty-tensor-to-alloc-tensor --one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map allow-return-allocs" --cse --canonicalize --vectorize-ondsp-fixed-memref-reduce="vector-width=4" --normalize-ondsp-fixed-vector-reduce > %t.vector.mlir
// RUN: FileCheck %s --check-prefix=VECTOR < %t.vector.mlir
// RUN: ondrix-opt %t.vector.mlir --convert-ondsp-fixed-to-scalar --expand-strided-metadata --lower-affine --convert-scf-to-cf --convert-vector-to-llvm --finalize-memref-to-llvm --convert-arith-to-llvm --convert-cf-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts > %t.vector-llvm.mlir
// RUN: ondrix-translate %t.vector-llvm.mlir --mlir-to-llvmir > %t.vector.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.vector.ll -o %t.vector.o
// RUN: cc -DFIR_DECIMATE_SYMBOL=_mlir_ciface_q15_fir_decimate -DFIR_DECIMATE_ACCUMULATOR_WIDTH=34 -DFIR_DECIMATE_UPDATE_OVERFLOW=WRAP -DFIR_DECIMATE_OUTPUT_ROUNDING=NEAREST_TIES_POSITIVE %S/Inputs/fir_decimate_q15_aot.c %t.vector.o -o %t.vector
// RUN: %t.vector

// SOURCE-LABEL: func.func @q15_fir_decimate
// SOURCE: ondrix.fir_decimate
// SOURCE-SAME: accumulator = !ondsp.acc<storage = i34, frac = 30, signed, update_overflow = wrap>
// SOURCE-SAME: factor = 2

// VECTOR-LABEL: func.func @q15_fir_decimate
// VECTOR: %[[FACTOR:.*]] = arith.constant 2 : index
// VECTOR: arith.muli {{.*}}, %[[FACTOR]] : index
// VECTOR-COUNT-2: vector.load {{.*}}vector<4xi16>
// VECTOR: arith.muli {{.*}} : vector<4xi32>
// VECTOR-COUNT-4: ondsp.acc_add_term
// VECTOR: ondsp.mac
// VECTOR-NOT: ondsp.reduce_mac
// VECTOR-NOT: ondrix.fir_decimate
