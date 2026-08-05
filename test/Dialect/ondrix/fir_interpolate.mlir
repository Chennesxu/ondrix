// RUN: ondrix-opt %s | FileCheck %s

// CHECK-LABEL: func.func @q15_interpolate_by_two
// CHECK: ondrix.fir_interpolate
// CHECK-SAME: factor = 2
// CHECK-SAME: numeric = #ondsp.fixed<signed, storage = i16, frac = 15>
func.func @q15_interpolate_by_two(
    %input: tensor<4xi16>, %coeffs: tensor<3xi16>, %init: tensor<9xi16>)
    -> tensor<9xi16> {
  %result = ondrix.fir_interpolate %input, %coeffs, %init {
    accumulator = !ondsp.acc<storage = i40, frac = 30, signed,
                              update_overflow = saturate>,
    dst = #ondsp.fixed<signed, storage = i16, frac = 15>,
    factor = 2,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    overflow = #ondsp.overflow<saturate>,
    product = #ondsp.product<full>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<4xi16>, tensor<3xi16>, tensor<9xi16>) -> tensor<9xi16>
  return %result : tensor<9xi16>
}

// CHECK-LABEL: func.func @f32_interpolate_by_two
// CHECK: ondrix.fir_interpolate
// CHECK-SAME: numeric = #ondsp.fp<format = f32, contract = off>
func.func @f32_interpolate_by_two(
    %input: tensor<4xf32>, %coeffs: tensor<3xf32>, %init: tensor<9xf32>)
    -> tensor<9xf32> {
  %result = ondrix.fir_interpolate %input, %coeffs, %init {
    factor = 2,
    numeric = #ondsp.fp<format = f32, contract = off>
  } : (tensor<4xf32>, tensor<3xf32>, tensor<9xf32>) -> tensor<9xf32>
  return %result : tensor<9xf32>
}

// CHECK-LABEL: func.func @q15_interpolate_ties_positive_export
// CHECK: ondrix.fir_interpolate
// CHECK-SAME: rounding = #ondsp.rounding<nearest_ties_positive>
func.func @q15_interpolate_ties_positive_export(
    %input: tensor<4xi16>, %coeffs: tensor<3xi16>, %init: tensor<9xi16>)
    -> tensor<9xi16> {
  %result = ondrix.fir_interpolate %input, %coeffs, %init {
    accumulator = !ondsp.acc<storage = i40, frac = 30, signed,
                              update_overflow = saturate>,
    dst = #ondsp.fixed<signed, storage = i16, frac = 15>,
    factor = 2,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    overflow = #ondsp.overflow<saturate>,
    product = #ondsp.product<full>,
    rounding = #ondsp.rounding<nearest_ties_positive>
  } : (tensor<4xi16>, tensor<3xi16>, tensor<9xi16>) -> tensor<9xi16>
  return %result : tensor<9xi16>
}
