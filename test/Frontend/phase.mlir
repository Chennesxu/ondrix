// RUN: ondrix-compile %S/Inputs/q15_phase.ox | FileCheck %s
// RUN: ondrix-compile %S/Inputs/q15_rfft_phase.ox | FileCheck %s --check-prefix=CHAIN
// RUN: ondrix-compile %S/Inputs/q31_phase.ox | FileCheck %s --check-prefix=Q31
// RUN: ondrix-compile %S/Inputs/q15_phase_turn32.ox | FileCheck %s --check-prefix=TURN32
// RUN: ondrix-compile %S/Inputs/q31_phase_turn32.ox | FileCheck %s --check-prefix=WIDE32
// RUN: not ondrix-compile %S/Inputs/invalid_phase_turn.ox 2>&1 | FileCheck %s --check-prefix=TURN
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

// The Q31 components select the packed-i32 profile; the turn reading stays Q0.16.
// Q31-LABEL: func.func @q31_phase(
// Q31-SAME: tensor<9xi64>) -> tensor<9xi16>
// Q31: ondrix.cx_phase
// Q31-SAME: layout = #ondsp.cx_layout<packed_i32_imag_hi_real_lo>
// Q31-SAME: numeric = #ondsp.fixed<signed, storage = i32, frac = 31>
// Q31-SAME: output_numeric = #ondsp.fixed<unsigned, storage = i16, frac = 16>

// ELEMENT: error: phase requires complex_q15 or complex_q31 operand elements

// The turn width is the one call-site choice, independent of the component
// width; q31 names the i32 storage of the Q0.32 turn the Q31 sine reads.
// TURN32-LABEL: func.func @q15_phase_turn32(
// TURN32-SAME: tensor<9xi32>) -> tensor<9xi32>
// TURN32: ondrix.cx_phase
// TURN32-SAME: layout = #ondsp.cx_layout<packed_i16_imag_hi_real_lo>
// TURN32-SAME: output_numeric = #ondsp.fixed<unsigned, storage = i32, frac = 32>
// WIDE32-LABEL: func.func @q31_phase_turn32(
// WIDE32-SAME: tensor<9xi64>) -> tensor<9xi32>
// WIDE32: ondrix.cx_phase
// WIDE32-SAME: output_numeric = #ondsp.fixed<unsigned, storage = i32, frac = 32>

// The contract admits exactly one tie rule, so there is no rounding
// parameter to accept, and the turn takes only the two widths that exist.
// ROUNDING: error: phase accepts only turn=q15 or turn=q31
// TURN: error: phase accepts only turn=q15 or turn=q31
