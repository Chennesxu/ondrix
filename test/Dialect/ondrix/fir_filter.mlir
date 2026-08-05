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

// CHECK-LABEL: func.func @q31_full_filter
// CHECK: ondrix.fir_filter
// CHECK-SAME: boundary = #ondrix.fir_boundary<full>
// CHECK-SAME: product = #ondsp.product<full>
func.func @q31_full_filter(
    %input: tensor<4xi32>, %coeffs: tensor<3xi32>, %init: tensor<6xi32>)
    -> tensor<6xi32> {
  %result = ondrix.fir_filter %input, %coeffs, %init {
    accumulator = !ondsp.acc<storage = i64, frac = 62, signed,
                              update_overflow = saturate>,
    boundary = #ondrix.fir_boundary<full>,
    dst = #ondsp.fixed<signed, storage = i32, frac = 31>,
    numeric = #ondsp.fixed<signed, storage = i32, frac = 31>,
    overflow = #ondsp.overflow<saturate>,
    product = #ondsp.product<full>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<4xi32>, tensor<3xi32>, tensor<6xi32>) -> tensor<6xi32>
  return %result : tensor<6xi32>
}

// CHECK-LABEL: func.func @q15_full_output_tile
// CHECK: %[[ORIGIN:.*]] = arith.constant 2 : index
// CHECK: ondrix.fir_filter %{{.*}}, %{{.*}}, %{{.*}}, %[[ORIGIN]]
// CHECK-SAME: boundary = #ondrix.fir_boundary<full>
func.func @q15_full_output_tile(
    %input: tensor<8xi16>, %coeffs: tensor<5xi16>, %init: tensor<4xi16>)
    -> tensor<4xi16> {
  %origin = arith.constant 2 : index
  %result = ondrix.fir_filter %input, %coeffs, %init, %origin {
    accumulator = !ondsp.acc<storage = i40, frac = 30, signed,
                              update_overflow = saturate>,
    boundary = #ondrix.fir_boundary<full>,
    dst = #ondsp.fixed<signed, storage = i16, frac = 15>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    overflow = #ondsp.overflow<saturate>,
    product = #ondsp.product<full>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<8xi16>, tensor<5xi16>, tensor<4xi16>, index) -> tensor<4xi16>
  return %result : tensor<4xi16>
}

// CHECK-LABEL: func.func @q15_fir_filter_ties_positive_export
// CHECK: ondrix.fir_filter
// CHECK-SAME: rounding = #ondsp.rounding<nearest_ties_positive>
func.func @q15_fir_filter_ties_positive_export(
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
    rounding = #ondsp.rounding<nearest_ties_positive>
  } : (tensor<8xi16>, tensor<3xi16>, tensor<6xi16>) -> tensor<6xi16>
  return %result : tensor<6xi16>
}
