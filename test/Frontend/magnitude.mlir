// RUN: ondrix-compile %S/Inputs/q15_magnitude.ox | FileCheck %s
// RUN: ondrix-compile %S/Inputs/q15_rfft_magnitude.ox | FileCheck %s --check-prefix=COMPOSE
// RUN: ondrix-compile %S/Inputs/q15_rfft_magnitude_floor.ox | FileCheck %s --check-prefix=FLOOR
// RUN: not ondrix-compile %S/Inputs/invalid_magnitude_element.ox 2>&1 | FileCheck %s --check-prefix=ELEMENT
// RUN: not ondrix-compile %S/Inputs/invalid_magnitude_result.ox 2>&1 | FileCheck %s --check-prefix=MISMATCH
// RUN: not ondrix-compile %S/Inputs/invalid_magnitude_rounding.ox 2>&1 | FileCheck %s --check-prefix=ROUNDING

// CHECK-LABEL: func.func @q15_magnitude(
// CHECK-SAME: %[[SPECTRUM:.*]]: tensor<9xi32>) -> tensor<9xi16>
// CHECK: %[[RESULT:.*]] = ondrix.cx_magnitude %[[SPECTRUM]]
// CHECK-SAME: layout = #ondsp.cx_layout<packed_i16_imag_hi_real_lo>
// CHECK-SAME: numeric = #ondsp.fixed<signed, storage = i16, frac = 15>
// CHECK-SAME: rounding = #ondsp.rounding<nearest_even>
// CHECK: return %[[RESULT]] : tensor<9xi16>

// COMPOSE-LABEL: func.func @q15_rfft_magnitude(
// COMPOSE-SAME: %[[INPUT:.*]]: tensor<16xi16>) -> tensor<9xi16>
// COMPOSE: %[[SPECTRUM:.*]] = ondrix.rfft %[[INPUT]]
// COMPOSE-SAME: (tensor<16xi16>) -> tensor<9xi32>
// COMPOSE: %[[MAGNITUDE:.*]] = ondrix.cx_magnitude %[[SPECTRUM]]
// COMPOSE-SAME: layout = #ondsp.cx_layout<packed_i16_imag_hi_real_lo>
// COMPOSE-SAME: rounding = #ondsp.rounding<nearest_even>
// COMPOSE-SAME: (tensor<9xi32>) -> tensor<9xi16>
// COMPOSE: return %[[MAGNITUDE]] : tensor<9xi16>

// The declared root rounding routes to the op attribute; the sum of squares
// stays exact inside the contract regardless of this choice.
// FLOOR-LABEL: func.func @q15_rfft_magnitude_floor(
// FLOOR: ondrix.rfft
// FLOOR: ondrix.cx_magnitude
// FLOOR-SAME: rounding = #ondsp.rounding<toward_negative>

// ELEMENT: invalid_magnitude_element.ox:2:10: error: magnitude requires complex_q15 operand elements
// MISMATCH: invalid_magnitude_result.ox:2:10: error: declared FFT result type does not match the builtin expression
// ROUNDING: invalid_magnitude_rounding.ox:2:10: error: magnitude root_rounding must be nearest_even or toward_negative
