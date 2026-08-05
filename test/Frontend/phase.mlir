// RUN: ondrix-compile %S/Inputs/q15_phase.ox | FileCheck %s
// RUN: ondrix-compile %S/Inputs/q15_rfft_phase.ox | FileCheck %s --check-prefix=CHAIN
// RUN: not ondrix-compile %S/Inputs/invalid_phase_element.ox 2>&1 | FileCheck %s --check-prefix=ELEMENT
// RUN: not ondrix-compile %S/Inputs/invalid_phase_rounding.ox 2>&1 | FileCheck %s --check-prefix=ROUNDING

// The result reading is the unsigned Q0.16 turn, supplied by the binding
// because the source type system names only the i16 storage.
// CHECK-LABEL: func.func @q15_phase(
// CHECK: ondrix.cx_phase
// CHECK-SAME: output_numeric = #ondsp.fixed<unsigned, storage = i16, frac = 16>
// CHECK-SAME: rounding = #ondsp.rounding<nearest_even>

// CHAIN-LABEL: func.func @q15_rfft_phase(
// CHAIN: ondrix.rfft
// CHAIN: ondrix.cx_phase

// ELEMENT: error: phase requires complex_q15 operand elements

// The contract admits exactly one tie rule, so there is no rounding
// parameter to accept — a spelled one is a parse error, not a narrowing.
// ROUNDING: error: expected ')' after phase operand
