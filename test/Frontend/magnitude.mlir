// RUN: ondrix-compile %S/Inputs/q15_magnitude.ox | FileCheck %s
// RUN: ondrix-compile %S/Inputs/q15_rfft_magnitude.ox | FileCheck %s --check-prefix=COMPOSE
// RUN: not ondrix-compile %S/Inputs/invalid_magnitude_element.ox 2>&1 | FileCheck %s --check-prefix=ELEMENT
// RUN: not ondrix-compile %S/Inputs/invalid_magnitude_result.ox 2>&1 | FileCheck %s --check-prefix=MISMATCH

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

// ELEMENT: invalid_magnitude_element.ox:2:10: error: magnitude requires complex_q15 operand elements
// MISMATCH: invalid_magnitude_result.ox:2:10: error: declared FFT result type does not match the builtin expression
