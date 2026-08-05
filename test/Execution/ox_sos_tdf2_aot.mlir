// RUN: ondrix-compile --emit=contracts %S/../Frontend/Inputs/f32_sos_tdf2.ox | ondrix-opt --ondrix-default-pipeline="vector-bits=256" > %t.mlir
// RUN: ondrix-translate %t.mlir --mlir-to-llvmir > %t.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.ll -o %t.o
// RUN: cc -ffp-contract=off %S/Inputs/ox_sos_tdf2_aot.c %t.o -lm -o %t
// RUN: %t

// The .ox source path to the two-section f32 TDF-II cascade under
// contract=off. Whole-chunk, split-chunk, and empty-chunk execution must all
// be bitwise equal to an independent recurrence written from the operation
// contract's event graph.
