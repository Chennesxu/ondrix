// RUN: ondrix-compile %S/Inputs/q15_filtered_spectrum.ox | FileCheck %s
// RUN: ondrix-compile %S/Inputs/q15_taps_spectrum.ox | FileCheck %s --check-prefix=COEFFS
// RUN: not ondrix-compile %S/Inputs/invalid_local_unused.ox 2>&1 | FileCheck %s --check-prefix=UNUSED
// RUN: not ondrix-compile %S/Inputs/invalid_local_reused.ox 2>&1 | FileCheck %s --check-prefix=REUSED
// RUN: not ondrix-compile %S/Inputs/invalid_local_collision.ox 2>&1 | FileCheck %s --check-prefix=COLLISION
// RUN: not ondrix-compile %S/Inputs/invalid_lowpass_return.ox 2>&1 | FileCheck %s --check-prefix=DESIGN-RETURN
// RUN: not ondrix-compile %S/Inputs/invalid_composed_boundary.ox 2>&1 | FileCheck %s --check-prefix=BOUNDARY
// RUN: not ondrix-compile %S/Inputs/invalid_lowpass_taps.ox 2>&1 | FileCheck %s --check-prefix=TAP-COUNT
// RUN: not ondrix-compile %S/Inputs/invalid_lowpass_cutoff.ox 2>&1 | FileCheck %s --check-prefix=CUTOFF
// RUN: not ondrix-compile %S/Inputs/invalid_composed_accumulator.ox 2>&1 | FileCheck %s --check-prefix=WIDTH
// RUN: not ondrix-compile %S/Inputs/invalid_unused_parameter.ox 2>&1 | FileCheck %s --check-prefix=PARAMETER

// The composed spectral program: local bindings name one builtin call each
// and are consumed exactly once by a later statement, so the checked kernel
// is the nested expression tree direct nesting would produce. The design
// stage is compile-time intent (evaluated later by the fail-closed
// quantization pass); the filter carries the executable Q15 export profile;
// the spectrum and magnitude stages keep their frozen packed-Q15 contracts.

// CHECK-LABEL: func.func @q15_filtered_spectrum(
// CHECK-SAME: %[[SIGNAL:.*]]: tensor<72xi16>) -> tensor<33xi16>
// CHECK: %[[TAPS:.*]] = ondrix.fir_design_windowed_sinc
// CHECK-SAME: cutoff_den = 4
// CHECK-SAME: cutoff_num = 1
// CHECK-SAME: numeric = #ondsp.fixed<signed, storage = i16, frac = 15>
// CHECK-SAME: response = #ondrix.fir_design_response<lowpass>
// CHECK-SAME: tensor<9xi16>
// CHECK: %[[INIT:.*]] = tensor.empty() : tensor<64xi16>
// CHECK: %[[FILTERED:.*]] = ondrix.fir_filter %[[SIGNAL]], %[[TAPS]], %[[INIT]]
// CHECK-SAME: accumulator = !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
// CHECK-SAME: boundary = #ondrix.fir_boundary<valid>
// CHECK-SAME: overflow = #ondsp.overflow<saturate>
// CHECK-SAME: rounding = #ondsp.rounding<nearest_even>
// CHECK-SAME: (tensor<72xi16>, tensor<9xi16>, tensor<64xi16>) -> tensor<64xi16>
// CHECK: %[[SPECTRUM:.*]] = ondrix.rfft %[[FILTERED]]
// CHECK-SAME: (tensor<64xi16>) -> tensor<33xi32>
// CHECK: %[[MAGNITUDE:.*]] = ondrix.cx_magnitude %[[SPECTRUM]]
// CHECK-SAME: (tensor<33xi32>) -> tensor<33xi16>
// CHECK: return %[[MAGNITUDE]] : tensor<33xi16>

// Coefficients may also arrive as a runtime tensor parameter, and the
// widened rfft extent set admits 32 points.
// COEFFS-LABEL: func.func @q15_taps_spectrum(
// COEFFS-SAME: %[[SIGNAL:.*]]: tensor<36xi16>, %[[TAPS:.*]]: tensor<5xi16>) -> tensor<17xi16>
// COEFFS: %[[FILTERED:.*]] = ondrix.fir_filter %[[SIGNAL]], %[[TAPS]],
// COEFFS: %[[SPECTRUM:.*]] = ondrix.rfft %[[FILTERED]]
// COEFFS-SAME: (tensor<32xi16>) -> tensor<17xi32>
// COEFFS: ondrix.cx_magnitude %[[SPECTRUM]]

// Every local binds exactly one use, and every parameter feeds the tree.
// UNUSED: invalid_local_unused.ox:2:3: error: local 'taps' is never consumed by a later statement
// REUSED: invalid_local_reused.ox:4:20: error: local 'filtered' is already consumed; each local binds exactly one use
// COLLISION: invalid_local_collision.ox:2:3: error: local 'signal' collides with an existing name
// PARAMETER: invalid_unused_parameter.ox:1:54: error: parameter 'extra' is never consumed by the kernel expression

// The design expression is not a kernel and the composed filter stage keeps
// its slice profile: valid boundary, explicit width-40 accumulator, odd tap
// count, and a strictly proper rational cutoff.
// DESIGN-RETURN: invalid_lowpass_return.ox:2:10: error: lowpass is a design expression; it is consumed by fir_filter coefficients
// BOUNDARY: invalid_composed_boundary.ox:3:14: error: a composed fir_filter currently supports boundary=valid
// TAP-COUNT: invalid_lowpass_taps.ox:2:10: error: lowpass currently requires an odd tap count in [3, 4095]
// CUTOFF: invalid_lowpass_cutoff.ox:2:10: error: lowpass cutoff must satisfy 0 < num/den < 1/2 strictly
// WIDTH: invalid_composed_accumulator.ox:3:14: error: the executable Q15 profile requires exact accumulator width 40
