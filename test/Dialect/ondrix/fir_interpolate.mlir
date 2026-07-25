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
