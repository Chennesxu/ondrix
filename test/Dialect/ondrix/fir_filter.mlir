// RUN: ondrix-opt %s | FileCheck %s

// CHECK-LABEL: func.func @q15_valid_filter
// CHECK: ondrix.fir_filter
// CHECK-SAME: accumulator = !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
// CHECK-SAME: boundary = #ondrix.fir_boundary<valid>
func.func @q15_valid_filter(
    %input: tensor<8xi16>, %coeffs: tensor<3xi16>, %init: tensor<6xi16>)
    -> tensor<6xi16> {
  %result = ondrix.fir_filter %input, %coeffs, %init {
    accumulator = !ondsp.acc<storage = i40, frac = 30, signed,
                              update_overflow = saturate>,
    boundary = #ondrix.fir_boundary<valid>,
    dst = #ondsp.fixed<signed, storage = i16, frac = 15>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    overflow = #ondsp.overflow<saturate>,
    product = #ondsp.product<full>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<8xi16>, tensor<3xi16>, tensor<6xi16>) -> tensor<6xi16>
  return %result : tensor<6xi16>
}

// CHECK-LABEL: func.func @f32_dynamic_valid_filter
// CHECK: ondrix.fir_filter
// CHECK-SAME: boundary = #ondrix.fir_boundary<valid>
// CHECK-SAME: numeric = #ondsp.fp<format = f32, contract = fma>
func.func @f32_dynamic_valid_filter(
    %input: tensor<?xf32>, %coeffs: tensor<?xf32>, %init: tensor<?xf32>)
    -> tensor<?xf32> {
  %result = ondrix.fir_filter %input, %coeffs, %init {
    boundary = #ondrix.fir_boundary<valid>,
    numeric = #ondsp.fp<format = f32, contract = fma>
  } : (tensor<?xf32>, tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  return %result : tensor<?xf32>
}
