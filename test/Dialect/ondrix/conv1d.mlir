// RUN: ondrix-opt %s | FileCheck %s

// CHECK-LABEL: func.func @q15_convolution
// CHECK: ondrix.conv1d
// CHECK-SAME: mode = #ondrix.conv1d_mode<convolution>
// CHECK-SAME: numeric = #ondsp.fixed<signed, storage = i16, frac = 15>
func.func @q15_convolution(
    %input: tensor<8xi16>, %kernel: tensor<3xi16>, %init: tensor<6xi16>)
    -> tensor<6xi16> {
  %result = ondrix.conv1d %input, %kernel, %init {
    accumulator = !ondsp.acc<storage = i40, frac = 30, signed,
                              update_overflow = saturate>,
    dst = #ondsp.fixed<signed, storage = i16, frac = 15>,
    mode = #ondrix.conv1d_mode<convolution>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    overflow = #ondsp.overflow<saturate>,
    product = #ondsp.product<full>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<8xi16>, tensor<3xi16>, tensor<6xi16>) -> tensor<6xi16>
  return %result : tensor<6xi16>
}

// CHECK-LABEL: func.func @f32_correlation
// CHECK: ondrix.conv1d
// CHECK-SAME: mode = #ondrix.conv1d_mode<correlation>
// CHECK-SAME: numeric = #ondsp.fp<format = f32, contract = fma>
func.func @f32_correlation(
    %input: tensor<?xf32>, %kernel: tensor<?xf32>, %init: tensor<?xf32>)
    -> tensor<?xf32> {
  %result = ondrix.conv1d %input, %kernel, %init {
    mode = #ondrix.conv1d_mode<correlation>,
    numeric = #ondsp.fp<format = f32, contract = fma>
  } : (tensor<?xf32>, tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  return %result : tensor<?xf32>
}
