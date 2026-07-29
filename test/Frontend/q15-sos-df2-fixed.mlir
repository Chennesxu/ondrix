// RUN: ondrix-compile %S/Inputs/q15_sos_df2_fixed.ox | FileCheck %s
// RUN: not ondrix-compile %S/Inputs/invalid_sos_state_ties_positive.ox 2>&1 | FileCheck %s --check-prefix=TIES

// Both recurrence boundaries expose only the three established tie rules.
// TIES: invalid_sos_state_ties_positive.ox:7:10: error: sos_df2_fixed state_rounding must be nearest_even, toward_negative, or toward_zero

// CHECK-LABEL: func.func @q15_sos_df2_fixed(
// CHECK-SAME: tensor<?xi16>, %{{.*}}: tensor<1x5xi16>, %{{.*}}: tensor<1xi16>,
// CHECK-SAME: tensor<1x2xi16>) -> (tensor<?xi16>, tensor<1x2xi16>)
// CHECK: ondrix.sos_filter_df2_fixed
// CHECK-SAME: accumulator = !ondsp.acc<storage = i40, frac = 30, signed,
// CHECK-SAME: update_overflow = saturate>
// CHECK-SAME: output_overflow = #ondsp.overflow<wrap>
// CHECK-SAME: output_rounding = #ondsp.rounding<toward_zero>
// CHECK-SAME: state_overflow = #ondsp.overflow<saturate>
// CHECK-SAME: state_rounding = #ondsp.rounding<nearest_even>
