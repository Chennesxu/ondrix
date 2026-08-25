// RUN: ondrix-compile %S/Inputs/f32_fir_filter_valid.ox --emit=llvm | FileCheck %s
// RUN: ondrix-compile %S/Inputs/q15_moving_average.ox --emit=llvm | FileCheck %s --check-prefix=Q15
// RUN: ondrix-compile %S/Inputs/q15_filtered_spectrum.ox --emit=llvm | FileCheck %s --check-prefix=SPECTRUM
// RUN: ondrix-compile %S/Inputs/q15_filtered_spectrum.ox --emit=llvm --vector-bits=128 | FileCheck %s --check-prefix=NARROW

// One tool invocation from .ox source to LLVM-dialect MLIR through the
// canonical pipeline. The user names numeric contracts in the source and
// nothing else: schedule candidates are generated, legality-filtered, and
// selected inside the compiler. Target facts are declared on the driver and
// assume nothing by default, so the plain invocations are ordered.

// CHECK: llvm.func @f32_fir_filter_valid
// CHECK-NOT: ondrix.
// CHECK-NOT: ondsp.

// Q15: llvm.func @q15_moving_average
// Q15-NOT: ondrix.

// The composed four-stage program (design, filter, spectrum, magnitude)
// rides the same single invocation: the design constants are evaluated, the
// staged spectrum is forwarded, and the filter is scheduled, all inside the
// canonical pipeline.
// SPECTRUM: llvm.func @q15_filtered_spectrum
// SPECTRUM-NOT: ondrix.
// SPECTRUM-NOT: ondsp.

// The driver's own target flags: an undeclared width is the ordered program,
// and at 128 bits the certified chunk is a whole number of the four-lane NEON
// and Helium vectors, here the two this reduction's extent can fill.
// CHECK-NOT: vector<
// SPECTRUM-NOT: vector<
// NARROW: vector<8xi64>
