// RUN: ondrix-compile --emit=contracts %S/../Frontend/Inputs/q15_window_spectrum.ox | ondrix-opt --ondrix-default-pipeline="vector-bits=256" > %t.mlir
// RUN: ondrix-translate %t.mlir --mlir-to-llvmir > %t.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.ll -o %t.o
// RUN: cc %S/Inputs/ox_window_design_aot.c %t.o -lm -o %t
// RUN: %t

// The design evaluation runs at the head of the canonical pipeline, so this
// pins that it fired before any object is built: an inert pass would leave
// the ondrix.window_hamming contract in place of the dense table.
// RUN: ondrix-compile --emit=contracts %S/../Frontend/Inputs/q15_window_spectrum.ox | ondrix-opt --evaluate-ondrix-fir-design | FileCheck %s --implicit-check-not="ondrix.window_hamming"
// CHECK-LABEL: func.func @q15_window_spectrum
// CHECK: dense<[2621, 7036, 17695, 28353, 32767, 28353, 17695, 7036, 2621]>

// A window design in the .ox coefficient slot, from source through the
// default pipeline to executed bits, against a reference that derives the
// window from its real-valued definition rather than reading the table back.
