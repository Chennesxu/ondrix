// RUN: ondrix-compile %S/Inputs/q15_sos_df2_fixed.ox | FileCheck %s
// RUN: ondrix-compile %S/Inputs/q15_sos_ties_positive.ox | FileCheck %s --check-prefix=TIES
// RUN: ondrix-compile %S/Inputs/q15_sos_df2_fixed_default_contract.ox | FileCheck %s --check-prefix=DEFAULT
// RUN: ondrix-compile %S/Inputs/q31_sos_df2_fixed.ox | FileCheck %s --check-prefix=Q31
// RUN: not ondrix-compile %S/Inputs/invalid_q31_sos_df2_fixed_default.ox 2>&1 | FileCheck %s --check-prefix=Q31-DEFAULT

// Omitting the whole policy takes the same rule the feed-forward default
// takes: three Q15 products bound a section sum by 3*2^30 < 2^39, so wrap
// is vacuous at i40, and both lossy exports take the export-default rule.
// DEFAULT-LABEL: func.func @q15_sos_df2_fixed_default_contract(
// DEFAULT: ondrix.sos_filter_df2_fixed
// DEFAULT-SAME: update_overflow = wrap>
// DEFAULT-SAME: output_overflow = #ondsp.overflow<saturate>
// DEFAULT-SAME: output_rounding = #ondsp.rounding<nearest_ties_positive>
// DEFAULT-SAME: state_overflow = #ondsp.overflow<saturate>
// DEFAULT-SAME: state_rounding = #ondsp.rounding<nearest_ties_positive>

// Both recurrence boundaries expose every declared tie rule they carry
// evidence for, independently.
// TIES-LABEL: func.func @q15_sos_ties_positive(
// TIES: ondrix.sos_filter_df2_fixed
// TIES-SAME: output_rounding = #ondsp.rounding<nearest_ties_positive>
// TIES-SAME: state_rounding = #ondsp.rounding<nearest_ties_positive>

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

// The Q31 section carries the op's i64/frac62 profile and spells every policy.
// Q31-LABEL: func.func @q31_sos_df2_fixed(
// Q31-SAME: tensor<?xi32>, %{{.*}}: tensor<1x5xi32>, %{{.*}}: tensor<1xi32>,
// Q31-SAME: tensor<1x2xi32>) -> (tensor<?xi32>, tensor<1x2xi32>)
// Q31: ondrix.sos_filter_df2_fixed
// Q31-SAME: accumulator = !ondsp.acc<storage = i64, frac = 62, signed,
// Q31-SAME: update_overflow = saturate>
// Q31-SAME: numeric = #ondsp.fixed<signed, storage = i32, frac = 31>
// Q31-SAME: output_rounding = #ondsp.rounding<toward_zero>

// Q31-DEFAULT: invalid_q31_sos_df2_fixed_default.ox:7:10: error: a Q31 sos_df2_fixed requires an explicit accumulator, state, and output policy
