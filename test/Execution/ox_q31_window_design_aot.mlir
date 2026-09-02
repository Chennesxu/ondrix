// RUN: ondrix-compile --emit=contracts %S/../Frontend/Inputs/q31_window_spectrum.ox | ondrix-opt --ondrix-default-pipeline="vector-bits=256" > %t.mlir
// RUN: ondrix-translate %t.mlir --mlir-to-llvmir > %t.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.ll -o %t.o
// RUN: cc %S/Inputs/ox_q31_window_design_aot.c %t.o -lm -o %t
// RUN: %t

// The Q31 design evaluates at the head of the canonical pipeline into the
// i32 table; the +1.0 center is the one saturated coefficient.
// RUN: ondrix-compile --emit=contracts %S/../Frontend/Inputs/q31_window_spectrum.ox | ondrix-opt --evaluate-ondrix-fir-design | FileCheck %s --implicit-check-not="ondrix.window_hamming"
// CHECK-LABEL: func.func @q31_window_spectrum
// CHECK: dense<[171798692, 461131055, 1159641170, 1858151285, 2147483647, 1858151285, 1159641170, 461131055, 171798692]>
