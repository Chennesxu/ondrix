// RUN: ondrix-compile %S/Inputs/q31_cfft8.ox | FileCheck %s
// RUN: ondrix-compile %S/Inputs/q31_icfft8.ox | FileCheck %s --check-prefix=INVERSE
// RUN: ondrix-compile %S/Inputs/q31_cfft_round_trip.ox | FileCheck %s --check-prefix=COMPOSE
// RUN: not ondrix-compile %S/Inputs/invalid_cfft_q31_rounding.ox 2>&1 | FileCheck %s --check-prefix=FROZEN
// RUN: not ondrix-compile %S/Inputs/invalid_cfft_q31_extent.ox 2>&1 | FileCheck %s --check-prefix=EXTENT
// RUN: not ondrix-compile %S/Inputs/invalid_q31_complex_consumer.ox 2>&1 | FileCheck %s --check-prefix=CONSUMER

// The binding emits the re-frozen packed-Q31 profile with no policy choice:
// raw-high per-term products, both stages toward_negative and saturating.
// CHECK-LABEL: func.func @q31_cfft8(
// CHECK-SAME: %[[INPUT:.*]]: tensor<8xi64>) -> tensor<8xi64>
// CHECK: %[[RESULT:.*]] = ondrix.cfft %[[INPUT]]
// CHECK-SAME: direction = #ondrix.cfft_direction<forward>
// CHECK-SAME: layout = #ondsp.cx_layout<packed_i32_imag_hi_real_lo>
// CHECK-SAME: numeric = #ondsp.fixed<signed, storage = i32, frac = 31>
// CHECK-SAME: output_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 1, rounding = toward_negative, overflow = saturate, saturate_to = i32>
// CHECK-SAME: product = #ondsp.product<high_raw>
// CHECK-SAME: product_scale = #ondsp.scale<pre_shift_left = 1, post_shift_right = 0, rounding = toward_negative, overflow = saturate, saturate_to = i32>
// CHECK: return %[[RESULT]] : tensor<8xi64>

// INVERSE-LABEL: func.func @q31_icfft8(
// INVERSE: %[[RESULT:.*]] = ondrix.cfft
// INVERSE-SAME: direction = #ondrix.cfft_direction<inverse>
// INVERSE-SAME: product = #ondsp.product<high_raw>
// INVERSE: return %[[RESULT]] : tensor<8xi64>

// The maximum Q31 extent composes, and the explicit frozen-policy spelling is
// accepted.
// COMPOSE-LABEL: func.func @q31_cfft_round_trip(
// COMPOSE-SAME: %[[INPUT:.*]]: tensor<64xi64>) -> tensor<64xi64>
// COMPOSE: %[[FORWARD:.*]] = ondrix.cfft %[[INPUT]]
// COMPOSE-SAME: direction = #ondrix.cfft_direction<forward>
// COMPOSE: %[[ROUND_TRIP:.*]] = ondrix.cfft %[[FORWARD]]
// COMPOSE-SAME: direction = #ondrix.cfft_direction<inverse>
// COMPOSE: return %[[ROUND_TRIP]] : tensor<64xi64>

// FROZEN: invalid_cfft_q31_rounding.ox:2:10: error: the packed-Q31 cfft profile is frozen to toward_negative saturating stages
// EXTENT: invalid_cfft_q31_extent.ox:2:10: error: a complex_q31 cfft currently supports power-of-two extents in [4, 64]
// CONSUMER: invalid_q31_complex_consumer.ox:2:10: error: magnitude requires complex_q15 operand elements
