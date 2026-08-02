// RUN: ondrix-compile --emit=contracts %S/../Frontend/Inputs/q15_filtered_spectrum.ox | ondrix-opt --ondrix-default-pipeline > %t.mlir
// RUN: ondrix-translate %t.mlir --mlir-to-llvmir > %t.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.ll -o %t.o
// RUN: cc %S/Inputs/composed_spectrum_q15_aot.c %t.o -o %t -lm
// RUN: %t

// RUN: ondrix-compile --emit=contracts %S/../Frontend/Inputs/q15_filtered_spectrum.ox | ondrix-opt --ondrix-default-pipeline="vector-bits=0" > %t.ordered.mlir
// RUN: ondrix-translate %t.ordered.mlir --mlir-to-llvmir > %t.ordered.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.ordered.ll -o %t.ordered.o
// RUN: cc %S/Inputs/composed_spectrum_q15_aot.c %t.ordered.o -o %t.ordered -lm
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
// RUN: ondrix-compile --emit=contracts %S/../Frontend/Inputs/q15_filtered_spectrum.ox | ondrix-opt --ondrix-default-pipeline | FileCheck %s --check-prefix=SCHEDULED
// SCHEDULED: llvm.func @q15_filtered_spectrum
// SCHEDULED: vector<8xi64>

// The composed four-stage spectral program from .ox source to executed bits:
// compile-time windowed-sinc lowpass design, valid-boundary FIR over the
// exact i40 saturating accumulator, the staged 64-point RFFT, and the
// magnitude spectrum. One tool invocation produces the module; the harness
// carries its own end-to-end reference — its own frozen golden tap table,
// its own quantized twiddles, its own correction-looped integer square root
// — so no stage is checked against itself.
//
// The gate doctrine: one independent reference, two schedule variants. The
// default width picks the certified vector schedules; width zero is the
// all-ordered program. Both objects run against the same reference and must
// agree with it bit for bit, on every one of the 33 bins. For exact
// fixed-point contracts the schedule cannot be allowed to move a numeric
// boundary, so schedule invariance is a checked fact here, not an assumption
// inherited from the transforms that produced the schedule.
