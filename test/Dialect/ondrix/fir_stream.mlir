// RUN: ondrix-opt %s | FileCheck %s

// CHECK-LABEL: func.func @q15_stream
// CHECK: ondrix.fir_stream
// CHECK-SAME: accumulator = !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
// CHECK-SAME: product = #ondsp.product<full>
func.func @q15_stream(
    %input: tensor<?xi16>, %coeffs: tensor<?xi16>, %state: tensor<?xi16>)
    -> (tensor<?xi16>, tensor<?xi16>) {
  %output, %next = ondrix.fir_stream %input, %coeffs, %state {
    accumulator = !ondsp.acc<storage = i40, frac = 30, signed,
                              update_overflow = saturate>,
    dst = #ondsp.fixed<signed, storage = i16, frac = 15>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    overflow = #ondsp.overflow<saturate>,
    product = #ondsp.product<full>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<?xi16>, tensor<?xi16>, tensor<?xi16>)
      -> (tensor<?xi16>, tensor<?xi16>)
  return %output, %next : tensor<?xi16>, tensor<?xi16>
}

// CHECK-LABEL: func.func @f32_stream
// CHECK: ondrix.fir_stream
// CHECK-SAME: numeric = #ondsp.fp<format = f32, contract = fma>
func.func @f32_stream(
    %input: tensor<4xf32>, %coeffs: tensor<3xf32>, %state: tensor<2xf32>)
    -> (tensor<4xf32>, tensor<2xf32>) {
  %output, %next = ondrix.fir_stream %input, %coeffs, %state {
    numeric = #ondsp.fp<format = f32, contract = fma>
  } : (tensor<4xf32>, tensor<3xf32>, tensor<2xf32>)
      -> (tensor<4xf32>, tensor<2xf32>)
  return %output, %next : tensor<4xf32>, tensor<2xf32>
}

// CHECK-LABEL: func.func @q15_fir_stream_ties_positive_export
// CHECK: ondrix.fir_stream
// CHECK-SAME: rounding = #ondsp.rounding<nearest_ties_positive>
func.func @q15_fir_stream_ties_positive_export(
    %input: tensor<4xi16>, %coeffs: tensor<3xi16>, %state: tensor<2xi16>)
    -> (tensor<4xi16>, tensor<2xi16>) {
  %output, %next = ondrix.fir_stream %input, %coeffs, %state {
    accumulator = !ondsp.acc<storage = i40, frac = 30, signed,
                              update_overflow = saturate>,
    dst = #ondsp.fixed<signed, storage = i16, frac = 15>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    overflow = #ondsp.overflow<saturate>,
    product = #ondsp.product<full>,
    rounding = #ondsp.rounding<nearest_ties_positive>
  } : (tensor<4xi16>, tensor<3xi16>, tensor<2xi16>)
      -> (tensor<4xi16>, tensor<2xi16>)
  return %output, %next : tensor<4xi16>, tensor<2xi16>
}
