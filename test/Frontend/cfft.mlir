// RUN: ondrix-compile %S/Inputs/q15_cfft8.ox | FileCheck %s
// RUN: ondrix-compile %S/Inputs/q15_icfft8.ox | FileCheck %s --check-prefix=INVERSE
// RUN: ondrix-compile %S/Inputs/q15_cfft_round_trip.ox | FileCheck %s --check-prefix=COMPOSE
// RUN: ondrix-compile %S/Inputs/q15_icfft_named_operand.ox | FileCheck %s --check-prefix=NAMED
// RUN: not ondrix-compile %S/Inputs/invalid_cfft_dynamic.ox 2>&1 | FileCheck %s --check-prefix=DYNAMIC
// RUN: not ondrix-compile %S/Inputs/invalid_cfft_extent.ox 2>&1 | FileCheck %s --check-prefix=EXTENT

// CHECK-LABEL: func.func @q15_cfft8(
// CHECK-SAME: %[[INPUT:.*]]: tensor<8xi32>) -> tensor<8xi32>
// CHECK: %[[RESULT:.*]] = ondrix.cfft %[[INPUT]]
// CHECK-SAME: direction = #ondrix.cfft_direction<forward>
// CHECK-SAME: layout = #ondsp.cx_layout<packed_i16_imag_hi_real_lo>
// CHECK-SAME: numeric = #ondsp.fixed<signed, storage = i16, frac = 15>
// CHECK-SAME: output_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 1, rounding = nearest_even, overflow = saturate, saturate_to = i16>
// CHECK-SAME: product = #ondsp.product<full>
// CHECK-SAME: product_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 15, rounding = nearest_even, overflow = saturate, saturate_to = i16>
// CHECK: return %[[RESULT]] : tensor<8xi32>

// INVERSE-LABEL: func.func @q15_icfft8(
// INVERSE-SAME: %[[INPUT:.*]]: tensor<8xi32>) -> tensor<8xi32>
// INVERSE: %[[RESULT:.*]] = ondrix.cfft %[[INPUT]]
// INVERSE-SAME: direction = #ondrix.cfft_direction<inverse>
// INVERSE-SAME: layout = #ondsp.cx_layout<packed_i16_imag_hi_real_lo>
// INVERSE-SAME: numeric = #ondsp.fixed<signed, storage = i16, frac = 15>
// INVERSE-SAME: output_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 1, rounding = nearest_even, overflow = saturate, saturate_to = i16>
// INVERSE-SAME: product = #ondsp.product<full>
// INVERSE-SAME: product_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 15, rounding = nearest_even, overflow = saturate, saturate_to = i16>
// INVERSE: return %[[RESULT]] : tensor<8xi32>

// COMPOSE-LABEL: func.func @q15_cfft_round_trip(
// COMPOSE-SAME: %[[INPUT:.*]]: tensor<8xi32>) -> tensor<8xi32>
// COMPOSE: %[[FORWARD:.*]] = ondrix.cfft %[[INPUT]]
// COMPOSE-SAME: direction = #ondrix.cfft_direction<forward>
// COMPOSE: %[[ROUND_TRIP:.*]] = ondrix.cfft %[[FORWARD]]
// COMPOSE-SAME: direction = #ondrix.cfft_direction<inverse>
// COMPOSE: return %[[ROUND_TRIP]] : tensor<8xi32>

// NAMED-LABEL: func.func @q15_icfft_named_operand(
// NAMED-SAME: %[[INPUT:.*]]: tensor<8xi32>) -> tensor<8xi32>
// NAMED: %[[RESULT:.*]] = ondrix.cfft %[[INPUT]]
// NAMED-SAME: direction = #ondrix.cfft_direction<inverse>
// NAMED: return %[[RESULT]] : tensor<8xi32>

// DYNAMIC: invalid_cfft_dynamic.ox:2:15: error: composable builtins currently require static operand extents
// EXTENT: invalid_cfft_extent.ox:2:10: error: cfft currently supports only four or eight points

// RUN: ondrix-compile %S/Inputs/q15_cfft8_target_profile.ox | FileCheck %s --check-prefix=TARGET
// RUN: ondrix-compile %S/Inputs/q15_icfft8_ntp.ox | FileCheck %s --check-prefix=NTP
// RUN: not ondrix-compile %S/Inputs/invalid_cfft_ne_wrap.ox 2>&1 | FileCheck %s --check-prefix=NEWRAP
// RUN: not ondrix-compile %S/Inputs/invalid_cfft_rounding.ox 2>&1 | FileCheck %s --check-prefix=BADROUND

// One declared pair names both scale boundaries, the gated inventory shape.
// TARGET-LABEL: func.func @q15_cfft8_floor(
// TARGET: ondrix.cfft
// TARGET-SAME: output_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 1, rounding = toward_negative, overflow = wrap, saturate_to = i16>
// TARGET-SAME: product_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 15, rounding = toward_negative, overflow = wrap, saturate_to = i16>

// NTP-LABEL: func.func @q15_icfft8_ntp(
// NTP: ondrix.cfft
// NTP-SAME: direction = #ondrix.cfft_direction<inverse>
// NTP-SAME: output_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 1, rounding = nearest_ties_positive, overflow = saturate, saturate_to = i16>
// NTP-SAME: product_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 15, rounding = nearest_ties_positive, overflow = saturate, saturate_to = i16>

// NEWRAP: a nearest_even cfft requires saturating overflow

// BADROUND: cfft rounding must be nearest_even, toward_negative, or nearest_ties_positive
