// RUN: ondrix-compile %S/Inputs/q15_rfft16.ox | FileCheck %s
// RUN: ondrix-compile %S/Inputs/q15_irfft16.ox | FileCheck %s --check-prefix=INVERSE
// RUN: ondrix-compile %S/Inputs/q15_rfft_round_trip.ox | FileCheck %s --check-prefix=COMPOSE
// RUN: not ondrix-compile %S/Inputs/invalid_rfft_dynamic.ox 2>&1 | FileCheck %s --check-prefix=DYNAMIC
// RUN: not ondrix-compile %S/Inputs/invalid_rfft_extent.ox 2>&1 | FileCheck %s --check-prefix=EXTENT
// RUN: not ondrix-compile %S/Inputs/invalid_rfft_element.ox 2>&1 | FileCheck %s --check-prefix=ELEMENT
// RUN: not ondrix-compile %S/Inputs/invalid_rfft_nesting.ox 2>&1 | FileCheck %s --check-prefix=NESTING

// CHECK-LABEL: func.func @q15_rfft16(
// CHECK-SAME: %[[INPUT:.*]]: tensor<16xi16>) -> tensor<9xi32>
// CHECK: %[[RESULT:.*]] = ondrix.rfft %[[INPUT]]
// CHECK-SAME: layout = #ondsp.cx_layout<packed_i16_imag_hi_real_lo>
// CHECK-SAME: numeric = #ondsp.fixed<signed, storage = i16, frac = 15>
// CHECK-SAME: output_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 1, rounding = nearest_even, overflow = saturate, saturate_to = i16>
// CHECK-SAME: product = #ondsp.product<full>
// CHECK-SAME: product_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 15, rounding = nearest_even, overflow = saturate, saturate_to = i16>
// CHECK: return %[[RESULT]] : tensor<9xi32>

// INVERSE-LABEL: func.func @q15_irfft16(
// INVERSE-SAME: %[[INPUT:.*]]: tensor<9xi32>) -> tensor<16xi16>
// INVERSE: %[[RESULT:.*]] = ondrix.irfft %[[INPUT]]
// INVERSE-SAME: layout = #ondsp.cx_layout<packed_i16_imag_hi_real_lo>
// INVERSE-SAME: numeric = #ondsp.fixed<signed, storage = i16, frac = 15>
// INVERSE-SAME: output_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 1, rounding = nearest_even, overflow = saturate, saturate_to = i16>
// INVERSE-SAME: product = #ondsp.product<full>
// INVERSE-SAME: product_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 15, rounding = nearest_even, overflow = saturate, saturate_to = i16>
// INVERSE: return %[[RESULT]] : tensor<16xi16>

// COMPOSE-LABEL: func.func @q15_rfft_round_trip(
// COMPOSE-SAME: %[[INPUT:.*]]: tensor<16xi16>) -> tensor<16xi16>
// COMPOSE: %[[SPECTRUM:.*]] = ondrix.rfft %[[INPUT]]
// COMPOSE: %[[ROUND_TRIP:.*]] = ondrix.irfft %[[SPECTRUM]]
// COMPOSE: return %[[ROUND_TRIP]] : tensor<16xi16>

// DYNAMIC: invalid_rfft_dynamic.ox:2:15: error: composable builtins currently require static operand extents
// EXTENT: invalid_rfft_extent.ox:2:10: error: rfft currently supports power-of-two extents in [8, 64]
// ELEMENT: invalid_rfft_element.ox:2:10: error: rfft requires Q15 real operand elements
// NESTING: invalid_rfft_nesting.ox:2:10: error: cfft currently supports only four or eight points
