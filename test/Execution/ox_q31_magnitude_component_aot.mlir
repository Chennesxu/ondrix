// RUN: ondrix-compile --emit=llvm %S/../Frontend/Inputs/q31_magnitude_component_even.ox --vector-bits=256 > %t.even.mlir
// RUN: ondrix-translate %t.even.mlir --mlir-to-llvmir > %t.even.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.even.ll -o %t.even.o
// RUN: ondrix-compile --emit=llvm %S/../Frontend/Inputs/q31_magnitude_component_floor.ox --vector-bits=256 > %t.floor.mlir
// RUN: ondrix-translate %t.floor.mlir --mlir-to-llvmir > %t.floor.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.floor.ll -o %t.floor.o
// RUN: cc %S/Inputs/ox_q31_magnitude_component_aot.c %t.even.o %t.floor.o -lm -o %t
// RUN: %t

// The Q31 component pre-shift is a per-call-site choice, so the source
// binding has to carry it all the way to two objects that disagree.
