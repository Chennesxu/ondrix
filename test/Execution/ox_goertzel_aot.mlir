// RUN: ondrix-compile --emit=contracts %S/../Frontend/Inputs/f32_goertzel_off.ox | ondrix-opt --ondrix-default-pipeline="vector-bits=256" > %t.off.mlir
// RUN: ondrix-translate %t.off.mlir --mlir-to-llvmir > %t.off.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.off.ll -o %t.off.o
// RUN: ondrix-compile --emit=contracts %S/../Frontend/Inputs/f32_goertzel_fma.ox | ondrix-opt --ondrix-default-pipeline="vector-bits=256" > %t.fma.mlir
// RUN: ondrix-translate %t.fma.mlir --mlir-to-llvmir > %t.fma.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.fma.ll -o %t.fma.o
// RUN: cc -ffp-contract=off %S/Inputs/ox_goertzel_aot.c %t.off.o %t.fma.o -lm -o %t
// RUN: %t

// The .ox source path to the f32 Goertzel contract. Both objects share the
// reference the dialect-sourced gate uses, so the source binding is checked
// against the same declared event graph rather than against itself.
