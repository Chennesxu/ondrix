// RUN: ondrix-compile --emit=contracts %S/../Frontend/Inputs/q15_filtered_spectrum.ox | ondrix-opt --ondrix-default-pipeline="vector-bits=256" > %t.mlir
// RUN: ondrix-translate %t.mlir --mlir-to-llvmir > %t.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.ll -o %t.o
// RUN: ondrix-compile --emit=contracts %S/../Frontend/Inputs/q15_filtered_spectrum_stage.ox | ondrix-opt --ondrix-default-pipeline="vector-bits=256" > %t.stage.mlir
// RUN: ondrix-translate %t.stage.mlir --mlir-to-llvmir > %t.stage.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.stage.ll -o %t.stage.o
// RUN: cc %S/Inputs/composed_spectrum_q15_aot.c %t.o %t.stage.o -o %t -lm
// RUN: %t

// RUN: ondrix-compile --emit=contracts %S/../Frontend/Inputs/q15_filtered_spectrum.ox | ondrix-opt --ondrix-default-pipeline="vector-bits=0" > %t.ordered.mlir
// RUN: ondrix-translate %t.ordered.mlir --mlir-to-llvmir > %t.ordered.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.ordered.ll -o %t.ordered.o
// RUN: ondrix-compile --emit=contracts %S/../Frontend/Inputs/q15_filtered_spectrum_stage.ox | ondrix-opt --ondrix-default-pipeline="vector-bits=0" > %t.ordered.stage.mlir
// RUN: ondrix-translate %t.ordered.stage.mlir --mlir-to-llvmir > %t.ordered.stage.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.ordered.stage.ll -o %t.ordered.stage.o
// RUN: cc %S/Inputs/composed_spectrum_q15_aot.c %t.ordered.o %t.ordered.stage.o -o %t.ordered -lm
// RUN: %t.ordered

// The forwarding attestation is self-attesting in the same sense as the
// rfft64 gate: it pins that the design evaluation and the staged-read
// forwarding actually fired at this extent, before any object is built. An
// inert forwarding pass would leave the packed 33xi32 intermediate alive,
// and an inert design evaluation would leave the ondrix.fir_design_*
// contract in place of the dense table below.
// RUN: ondrix-compile --emit=contracts %S/../Frontend/Inputs/q15_filtered_spectrum.ox | ondrix-opt --evaluate-ondrix-fir-design --convert-ondrix-to-ondsp="preserve-bufferizable-reductions=true" --canonicalize --cse --forward-ondrix-insert-extract --canonicalize --cse | FileCheck %s --check-prefix=FORWARDED --implicit-check-not="tensor<33xi32>"
// FORWARDED-LABEL: func.func @q15_filtered_spectrum
// FORWARDED: dense<[0, -747, 0, 9025, 16384, 9025, 0, -747, 0]>

// The schedule attestation pins that the certified saturating horizontal
// reduction really fired inside the canonical pipeline: the executed default
// object below is a vectorized schedule, not a scalar one that happens to
// agree with the reference.
// RUN: ondrix-compile --emit=contracts %S/../Frontend/Inputs/q15_filtered_spectrum.ox | ondrix-opt --ondrix-default-pipeline="vector-bits=256" | FileCheck %s --check-prefix=SCHEDULED
// SCHEDULED: llvm.func @q15_filtered_spectrum
// SCHEDULED: vector<8xi64>

// The composed four-stage spectral program from .ox source to executed bits:
// compile-time windowed-sinc lowpass design, valid-boundary FIR over the
// exact i40 saturating accumulator, the staged 64-point RFFT, and the
// magnitude spectrum. The harness carries its own end-to-end reference —
// frozen golden tap table, quantized twiddles, correction-looped integer
// square root — so no stage is checked against itself. Both schedule
// variants (default width, ordered width zero) must match it bit for bit on
// every bin: schedule invariance of the exact contracts is checked, not
// assumed.
