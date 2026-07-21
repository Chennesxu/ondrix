// RUN: ondrix-opt %s | FileCheck %s

// CHECK-LABEL: func.func @q15_dynamic
// CHECK: ondrix.sos_filter_df2_fixed
// CHECK-SAME: accumulator = !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
// CHECK-SAME: output_overflow = #ondsp.overflow<wrap>
// CHECK-SAME: output_rounding = #ondsp.rounding<toward_zero>
// CHECK-SAME: state_overflow = #ondsp.overflow<saturate>
// CHECK-SAME: state_rounding = #ondsp.rounding<nearest_even>
func.func @q15_dynamic(
    %input: tensor<?xi16>, %coeffs: tensor<?x5xi16>,
    %scales: tensor<?xi16>, %state: tensor<?x2xi16>)
    -> (tensor<?xi16>, tensor<?x2xi16>) {
  %output, %next = ondrix.sos_filter_df2_fixed %input, %coeffs, %scales, %state {
    accumulator = !ondsp.acc<storage = i40, frac = 30, signed,
                              update_overflow = saturate>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    output_overflow = #ondsp.overflow<wrap>,
    output_rounding = #ondsp.rounding<toward_zero>,
    product = #ondsp.product<full>,
    state_overflow = #ondsp.overflow<saturate>,
    state_rounding = #ondsp.rounding<nearest_even>
  } : (tensor<?xi16>, tensor<?x5xi16>, tensor<?xi16>, tensor<?x2xi16>)
      -> (tensor<?xi16>, tensor<?x2xi16>)
  return %output, %next : tensor<?xi16>, tensor<?x2xi16>
}

// CHECK-LABEL: func.func @q31_static
// CHECK: ondrix.sos_filter_df2_fixed
// CHECK-SAME: accumulator = !ondsp.acc<storage = i64, frac = 62, signed, update_overflow = wrap>
// CHECK-SAME: numeric = #ondsp.fixed<signed, storage = i32, frac = 31>
func.func @q31_static(
    %input: tensor<8xi32>, %coeffs: tensor<2x5xi32>,
    %scales: tensor<2xi32>, %state: tensor<2x2xi32>)
    -> (tensor<8xi32>, tensor<2x2xi32>) {
  %output, %next = ondrix.sos_filter_df2_fixed %input, %coeffs, %scales, %state {
    accumulator = !ondsp.acc<storage = i64, frac = 62, signed,
                              update_overflow = wrap>,
    numeric = #ondsp.fixed<signed, storage = i32, frac = 31>,
    output_overflow = #ondsp.overflow<saturate>,
    output_rounding = #ondsp.rounding<toward_negative>,
    product = #ondsp.product<full>,
    state_overflow = #ondsp.overflow<wrap>,
    state_rounding = #ondsp.rounding<toward_zero>
  } : (tensor<8xi32>, tensor<2x5xi32>, tensor<2xi32>, tensor<2x2xi32>)
      -> (tensor<8xi32>, tensor<2x2xi32>)
  return %output, %next : tensor<8xi32>, tensor<2x2xi32>
}
