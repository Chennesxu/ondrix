// RUN: ondrix-opt %s --convert-ondrix-to-ondsp | FileCheck %s

// CHECK-LABEL: func.func @q15_dynamic
// CHECK-COUNT-2: cf.assert
// CHECK: %[[SAMPLES:.*]]:2 = scf.for
// CHECK: %[[SECTIONS:.*]]:2 = scf.for
// CHECK-COUNT-8: tensor.extract
// CHECK: %[[STATE_ZERO:.*]] = ondsp.acc_zero
// CHECK: %[[STATE1:.*]] = ondsp.mac %[[STATE_ZERO]]
// CHECK: %[[STATE2:.*]] = ondsp.mac %[[STATE1]]
// CHECK: %[[STATE3:.*]] = ondsp.mac %[[STATE2]]
// CHECK: %[[D1:.*]] = ondsp.acc_export %[[STATE3]]
// CHECK-SAME: overflow = #ondsp.overflow<saturate>
// CHECK-SAME: rounding = #ondsp.rounding<nearest_even>
// CHECK: %[[OUTPUT_ZERO:.*]] = ondsp.acc_zero
// CHECK: %[[OUTPUT1:.*]] = ondsp.mac %[[OUTPUT_ZERO]]
// CHECK: %[[OUTPUT2:.*]] = ondsp.mac %[[OUTPUT1]]
// CHECK: %[[OUTPUT3:.*]] = ondsp.mac %[[OUTPUT2]]
// CHECK: %[[OUTPUT:.*]] = ondsp.acc_export %[[OUTPUT3]]
// CHECK-SAME: overflow = #ondsp.overflow<wrap>
// CHECK-SAME: rounding = #ondsp.rounding<toward_zero>
// CHECK: tensor.insert %[[D1]]
// CHECK: tensor.insert %{{.*}}
// CHECK: scf.yield %[[OUTPUT]]
// CHECK: tensor.insert %[[SECTIONS]]#0
// CHECK: return %[[SAMPLES]]#0, %[[SAMPLES]]#1
// CHECK-NOT: ondrix.sos_filter_df2_fixed
func.func @q15_dynamic(
    %input: tensor<?xi16>, %coeffs: tensor<?x5xi16>,
    %scales: tensor<?xi16>, %state: tensor<?x2xi16>)
    -> (tensor<?xi16>, tensor<?x2xi16>) {
  %output, %next = ondrix.sos_filter_df2_fixed %input, %coeffs, %scales, %state {
    accumulator = !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    output_overflow = #ondsp.overflow<wrap>, output_rounding = #ondsp.rounding<toward_zero>,
    product = #ondsp.product<full>, state_overflow = #ondsp.overflow<saturate>,
    state_rounding = #ondsp.rounding<nearest_even>
  } : (tensor<?xi16>, tensor<?x5xi16>, tensor<?xi16>, tensor<?x2xi16>)
      -> (tensor<?xi16>, tensor<?x2xi16>)
  return %output, %next : tensor<?xi16>, tensor<?x2xi16>
}

// CHECK-LABEL: func.func @q31_static
// CHECK: ondsp.acc_zero : <storage = i64, frac = 62, signed, update_overflow = wrap>
// CHECK-COUNT-3: ondsp.mac
// CHECK: ondsp.acc_export
// CHECK: ondsp.acc_zero
// CHECK-COUNT-3: ondsp.mac
// CHECK: ondsp.acc_export
// CHECK-NOT: ondrix.sos_filter_df2_fixed
func.func @q31_static(
    %input: tensor<4xi32>, %coeffs: tensor<2x5xi32>,
    %scales: tensor<2xi32>, %state: tensor<2x2xi32>)
    -> (tensor<4xi32>, tensor<2x2xi32>) {
  %output, %next = ondrix.sos_filter_df2_fixed %input, %coeffs, %scales, %state {
    accumulator = !ondsp.acc<storage = i64, frac = 62, signed, update_overflow = wrap>,
    numeric = #ondsp.fixed<signed, storage = i32, frac = 31>,
    output_overflow = #ondsp.overflow<saturate>,
    output_rounding = #ondsp.rounding<toward_negative>, product = #ondsp.product<full>,
    state_overflow = #ondsp.overflow<wrap>, state_rounding = #ondsp.rounding<toward_zero>
  } : (tensor<4xi32>, tensor<2x5xi32>, tensor<2xi32>, tensor<2x2xi32>)
      -> (tensor<4xi32>, tensor<2x2xi32>)
  return %output, %next : tensor<4xi32>, tensor<2x2xi32>
}
