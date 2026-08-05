// RUN: ondrix-compile %S/Inputs/q15_sos_df2_fixed.ox | FileCheck %s
// RUN: ondrix-compile %S/Inputs/q15_sos_ties_positive.ox | FileCheck %s --check-prefix=TIES
// RUN: ondrix-compile %S/Inputs/q15_sos_df2_fixed_default_contract.ox | FileCheck %s --check-prefix=DEFAULT

// Omitting the whole policy takes the same rule the feed-forward default
// takes: three Q15 products bound a section sum by 3*2^30 < 2^39, so wrap
// is vacuous at i40, and both exports lose information and round unbiased.
// DEFAULT-LABEL: func.func @q15_sos_df2_fixed_default_contract(
// DEFAULT: ondrix.sos_filter_df2_fixed
// DEFAULT-SAME: update_overflow = wrap>
// DEFAULT-SAME: output_overflow = #ondsp.overflow<saturate>
// DEFAULT-SAME: output_rounding = #ondsp.rounding<nearest_even>
// DEFAULT-SAME: state_overflow = #ondsp.overflow<saturate>
// DEFAULT-SAME: state_rounding = #ondsp.rounding<nearest_even>

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
