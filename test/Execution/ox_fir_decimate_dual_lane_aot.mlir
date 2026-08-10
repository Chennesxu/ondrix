// RUN: ondrix-compile %S/../Frontend/Inputs/q15_fir_decimate.ox > %t.source.mlir
// RUN: ondrix-opt %t.source.mlir --empty-tensor-to-alloc-tensor --one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map allow-return-allocs" --cse --canonicalize --widen-ondsp-exact-accumulators --vectorize-ondsp-fixed-decimate-outputs="vector-width=2" --scalarize-ondsp-fixed-reduce-mac --canonicalize --convert-ondsp-lane-pairs-to-ortumcore --convert-ondsp-to-ortumcore > %t.ortumcore.mlir
// RUN: FileCheck %s --check-prefix=DUAL --implicit-check-not=ondsp. < %t.ortumcore.mlir
// RUN: ondrix-opt %t.ortumcore.mlir --convert-ortumcore-to-ondsp-emulation --convert-ondsp-fixed-to-scalar --expand-strided-metadata --lower-affine --convert-scf-to-cf --convert-vector-to-llvm --finalize-memref-to-llvm --convert-arith-to-llvm --convert-cf-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts > %t.target.mlir
// RUN: ondrix-translate %t.target.mlir --mlir-to-llvmir > %t.target.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.target.ll -o %t.target.o
// RUN: cc -DFIR_DECIMATE_SYMBOL=_mlir_ciface_q15_fir_decimate -DFIR_DECIMATE_TIES_POSITIVE_SYMBOL=_mlir_ciface_q15_fir_decimate -DFIR_DECIMATE_ACCUMULATOR_WIDTH=34 -DFIR_DECIMATE_UPDATE_OVERFLOW=WRAP -DFIR_DECIMATE_OUTPUT_ROUNDING=NEAREST_TIES_POSITIVE %S/Inputs/fir_decimate_q15_aot.c %t.target.o -o %t.target
// RUN: %t.target

// The whole promise chain for the fourth capability, from an unannotated
// source: the default decimating FIR reaches the dual-lane MAC.

// The batched two output lanes advance through one dmac per tap with the
// broadcast coefficient in both multiplier halves; the exactness widening
// makes the declared i34 wrap web target-typed first, so both exports run
// the proven shift-14 ties-positive readout composition per lane.
// DUAL-LABEL: func.func @q15_fir_decimate(
// DUAL-COUNT-2: ortumcore.acc_init
// DUAL: %[[OUT:.*]], %[[OUT2:.*]] = ortumcore.dmac {{.*}} : (!ortumcore.acc, !ortumcore.acc, i16, i16, i16, i16)
// DUAL-COUNT-4: ortumcore.dmac
// DUAL: ortumcore.acc_out %{{.*}} {shift = 14
// DUAL: ortumcore.acc_out %{{.*}} {shift = 14
// The uncovered remainder outputs keep the ordered schedule: the scalarized
// reduction is a plain scalar MAC loop with the same readout composition.
// DUAL: scf.for
// DUAL: ortumcore.mac_add
// DUAL: ortumcore.acc_out %{{.*}} {shift = 14
