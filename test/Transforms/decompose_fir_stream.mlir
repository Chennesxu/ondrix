// RUN: ondrix-opt %s --decompose-ondrix-fir-stream | FileCheck %s

// CHECK-LABEL: func.func @q15_stream
// CHECK-COUNT-3: cf.assert
// CHECK: %[[EXTENDED:.*]] = tensor.empty
// CHECK: %[[WITH_STATE:.*]] = tensor.insert_slice %{{.*}} into %[[EXTENDED]][0]
// CHECK: %[[SEQUENCE:.*]] = tensor.insert_slice %{{.*}} into %[[WITH_STATE]]
// CHECK: %[[HAS_INPUT:.*]] = arith.cmpi ugt
// CHECK: %[[OUTPUT:.*]] = scf.if %[[HAS_INPUT]] -> (tensor<?xi16>) {
// CHECK: ondrix.fir_filter %[[SEQUENCE]]
// CHECK-SAME: boundary = #ondrix.fir_boundary<valid>
// CHECK: } else {
// CHECK: tensor.extract_slice %[[SEQUENCE]]
// CHECK-NOT: ondrix.fir_stream
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

// CHECK-LABEL: func.func @f32_static
// CHECK: scf.if
// CHECK: ondrix.fir_filter
// CHECK-SAME: numeric = #ondsp.fp<format = f32, contract = off>
// CHECK: tensor.extract_slice {{.*}} : tensor<?xf32> to tensor<2xf32>
func.func @f32_static(
    %input: tensor<4xf32>, %coeffs: tensor<3xf32>, %state: tensor<2xf32>)
    -> (tensor<4xf32>, tensor<2xf32>) {
  %output, %next = ondrix.fir_stream %input, %coeffs, %state {
    numeric = #ondsp.fp<format = f32, contract = off>
  } : (tensor<4xf32>, tensor<3xf32>, tensor<2xf32>)
      -> (tensor<4xf32>, tensor<2xf32>)
  return %output, %next : tensor<4xf32>, tensor<2xf32>
}
