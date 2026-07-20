// RUN: ondrix-opt %s --convert-ondrix-to-ondsp | FileCheck %s

// CHECK-LABEL: func.func @q15_filter
// CHECK: %[[C0:.*]] = arith.constant 0 : index
// CHECK: %[[C1:.*]] = arith.constant 1 : index
// CHECK: tensor.dim
// CHECK: cf.assert
// CHECK: %[[OUTER:.*]] = scf.for
// CHECK: %[[ZERO:.*]] = ondsp.acc_zero
// CHECK: %[[INNER:.*]] = scf.for
// CHECK: tensor.extract
// CHECK: tensor.extract
// CHECK: ondsp.mac
// CHECK: ondsp.acc_export
// CHECK: tensor.insert
// CHECK-NOT: ondrix.fir_filter
func.func @q15_filter(
    %input: tensor<?xi16>, %coeffs: tensor<?xi16>, %init: tensor<?xi16>)
    -> tensor<?xi16> {
  %result = ondrix.fir_filter %input, %coeffs, %init {
    accumulator = !ondsp.acc<storage = i40, frac = 30, signed,
                              update_overflow = saturate>,
    boundary = #ondrix.fir_boundary<valid>,
    dst = #ondsp.fixed<signed, storage = i16, frac = 15>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    overflow = #ondsp.overflow<saturate>,
    product = #ondsp.product<full>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<?xi16>, tensor<?xi16>, tensor<?xi16>) -> tensor<?xi16>
  return %result : tensor<?xi16>
}

// CHECK-LABEL: func.func @f32_filter
// CHECK: scf.for
// CHECK: arith.constant 0.000000e+00 : f32
// CHECK: scf.for
// CHECK: math.fma
// CHECK: tensor.insert
// CHECK-NOT: ondrix.fir_filter
func.func @f32_filter(
    %input: tensor<8xf32>, %coeffs: tensor<3xf32>, %init: tensor<6xf32>)
    -> tensor<6xf32> {
  %result = ondrix.fir_filter %input, %coeffs, %init {
    boundary = #ondrix.fir_boundary<valid>,
    numeric = #ondsp.fp<format = f32, contract = fma>
  } : (tensor<8xf32>, tensor<3xf32>, tensor<6xf32>) -> tensor<6xf32>
  return %result : tensor<6xf32>
}

// CHECK-LABEL: func.func @f32_filter_off
// CHECK: scf.for
// CHECK: arith.mulf
// CHECK: arith.addf
// CHECK-NOT: math.fma
// CHECK: tensor.insert
func.func @f32_filter_off(
    %input: tensor<8xf32>, %coeffs: tensor<3xf32>, %init: tensor<6xf32>)
    -> tensor<6xf32> {
  %result = ondrix.fir_filter %input, %coeffs, %init {
    boundary = #ondrix.fir_boundary<valid>,
    numeric = #ondsp.fp<format = f32, contract = off>
  } : (tensor<8xf32>, tensor<3xf32>, tensor<6xf32>) -> tensor<6xf32>
  return %result : tensor<6xf32>
}

// CHECK-LABEL: func.func @f32_filter_fast
// CHECK: scf.for
// CHECK: math.fma {{.*}} fastmath<fast> : f32
// CHECK: tensor.insert
func.func @f32_filter_fast(
    %input: tensor<8xf32>, %coeffs: tensor<3xf32>, %init: tensor<6xf32>)
    -> tensor<6xf32> {
  %result = ondrix.fir_filter %input, %coeffs, %init {
    boundary = #ondrix.fir_boundary<valid>,
    numeric = #ondsp.fp<format = f32, contract = fast>
  } : (tensor<8xf32>, tensor<3xf32>, tensor<6xf32>) -> tensor<6xf32>
  return %result : tensor<6xf32>
}

// CHECK-LABEL: func.func @q15_full_filter
// CHECK-COUNT-3: cf.assert
// CHECK: scf.for
// CHECK: scf.for
// CHECK: arith.cmpi uge
// CHECK: arith.cmpi ult
// CHECK: scf.if
// CHECK: ondsp.mac
// CHECK: ondsp.acc_export
// CHECK-NOT: ondrix.fir_filter
func.func @q15_full_filter(
    %input: tensor<?xi16>, %coeffs: tensor<?xi16>, %init: tensor<?xi16>)
    -> tensor<?xi16> {
  %result = ondrix.fir_filter %input, %coeffs, %init {
    accumulator = !ondsp.acc<storage = i40, frac = 30, signed,
                              update_overflow = saturate>,
    boundary = #ondrix.fir_boundary<full>,
    dst = #ondsp.fixed<signed, storage = i16, frac = 15>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    overflow = #ondsp.overflow<saturate>,
    product = #ondsp.product<full>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<?xi16>, tensor<?xi16>, tensor<?xi16>) -> tensor<?xi16>
  return %result : tensor<?xi16>
}

// CHECK-LABEL: func.func @f32_full_filter
// CHECK: scf.if
// CHECK: math.fma
// CHECK: tensor.insert
// CHECK-NOT: ondrix.fir_filter
func.func @f32_full_filter(
    %input: tensor<2xf32>, %coeffs: tensor<3xf32>, %init: tensor<4xf32>)
    -> tensor<4xf32> {
  %result = ondrix.fir_filter %input, %coeffs, %init {
    boundary = #ondrix.fir_boundary<full>,
    numeric = #ondsp.fp<format = f32, contract = fma>
  } : (tensor<2xf32>, tensor<3xf32>, tensor<4xf32>) -> tensor<4xf32>
  return %result : tensor<4xf32>
}
