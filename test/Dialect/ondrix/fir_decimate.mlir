// RUN: ondrix-opt %s | FileCheck %s

// CHECK-LABEL: func.func @q15_decimate_by_two
// CHECK: ondrix.fir_decimate
// CHECK-SAME: factor = 2
// CHECK-SAME: numeric = #ondsp.fixed<signed, storage = i16, frac = 15>
func.func @q15_decimate_by_two(
    %input: tensor<12xi16>, %coeffs: tensor<5xi16>, %init: tensor<4xi16>)
    -> tensor<4xi16> {
  %result = ondrix.fir_decimate %input, %coeffs, %init {
    accumulator = !ondsp.acc<storage = i40, frac = 30, signed,
                              update_overflow = saturate>,
    dst = #ondsp.fixed<signed, storage = i16, frac = 15>,
    factor = 2,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    overflow = #ondsp.overflow<saturate>,
    product = #ondsp.product<full>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<12xi16>, tensor<5xi16>, tensor<4xi16>) -> tensor<4xi16>
  return %result : tensor<4xi16>
}
