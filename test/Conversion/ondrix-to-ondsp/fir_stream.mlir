// RUN: ondrix-opt %s --convert-ondrix-to-ondsp | FileCheck %s

// CHECK-LABEL: func.func @q15_stream
// CHECK-COUNT-3: cf.assert
// CHECK: %[[OUTPUT:.*]] = tensor.empty
// CHECK: %[[NEXT:.*]] = tensor.empty
// CHECK: %[[OUTPUT_LOOP:.*]] = scf.for
// CHECK: ondsp.acc_zero
// CHECK: scf.for
// CHECK: arith.addi
// CHECK: arith.cmpi ult
// CHECK: scf.if
// CHECK: tensor.extract
// CHECK: tensor.extract
// CHECK: ondsp.mac
// CHECK: ondsp.acc_export
// CHECK: tensor.insert
// CHECK: %[[STATE_LOOP:.*]] = scf.for
// CHECK: scf.if
// CHECK: tensor.insert
// CHECK: return %[[OUTPUT_LOOP]], %[[STATE_LOOP]]
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

// CHECK-LABEL: func.func @f32_stream
// CHECK: tensor.empty() : tensor<4xf32>
// CHECK: tensor.empty() : tensor<2xf32>
// CHECK: math.fma
// CHECK-NOT: ondrix.fir_stream
func.func @f32_stream(
    %input: tensor<4xf32>, %coeffs: tensor<3xf32>, %state: tensor<2xf32>)
    -> (tensor<4xf32>, tensor<2xf32>) {
  %output, %next = ondrix.fir_stream %input, %coeffs, %state {
    numeric = #ondsp.fp<format = f32, contract = fma>
  } : (tensor<4xf32>, tensor<3xf32>, tensor<2xf32>)
      -> (tensor<4xf32>, tensor<2xf32>)
  return %output, %next : tensor<4xf32>, tensor<2xf32>
}

// CHECK-LABEL: func.func @unit_tap_empty_state
// CHECK: tensor.empty() : tensor<0xi16>
// CHECK: scf.for
// CHECK-NOT: ondrix.fir_stream
func.func @unit_tap_empty_state(
    %input: tensor<3xi16>, %coeffs: tensor<1xi16>, %state: tensor<0xi16>)
    -> (tensor<3xi16>, tensor<0xi16>) {
  %output, %next = ondrix.fir_stream %input, %coeffs, %state {
    accumulator = !ondsp.acc<storage = i40, frac = 30, signed,
                              update_overflow = wrap>,
    dst = #ondsp.fixed<signed, storage = i16, frac = 15>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    overflow = #ondsp.overflow<wrap>,
    product = #ondsp.product<full>,
    rounding = #ondsp.rounding<toward_zero>
  } : (tensor<3xi16>, tensor<1xi16>, tensor<0xi16>)
      -> (tensor<3xi16>, tensor<0xi16>)
  return %output, %next : tensor<3xi16>, tensor<0xi16>
}

// CHECK-LABEL: func.func @mixed_static_results
// CHECK: cf.assert {{.*}}, "FIR stream output length must equal input chunk length"
// CHECK: cf.assert {{.*}}, "FIR stream next-state length must equal state length"
// CHECK: tensor.empty() : tensor<4xf32>
// CHECK: tensor.empty() : tensor<2xf32>
// CHECK-NOT: ondrix.fir_stream
func.func @mixed_static_results(
    %input: tensor<?xf32>, %coeffs: tensor<?xf32>, %state: tensor<?xf32>)
    -> (tensor<4xf32>, tensor<2xf32>) {
  %output, %next = ondrix.fir_stream %input, %coeffs, %state {
    numeric = #ondsp.fp<format = f32, contract = off>
  } : (tensor<?xf32>, tensor<?xf32>, tensor<?xf32>)
      -> (tensor<4xf32>, tensor<2xf32>)
  return %output, %next : tensor<4xf32>, tensor<2xf32>
}

// CHECK-LABEL: func.func @f32_stream_fast
// CHECK: math.fma %{{[^ ]*}}, %{{[^ ]*}}, %{{[^ ]*}} {{.*}}used_permissions = ["fuse_multiply_add"]{{.*}} : f32
// CHECK-NOT: ondrix.fir_stream
func.func @f32_stream_fast(
    %input: tensor<4xf32>, %coeffs: tensor<3xf32>, %state: tensor<2xf32>)
    -> (tensor<4xf32>, tensor<2xf32>) {
  %output, %next = ondrix.fir_stream %input, %coeffs, %state {
    numeric = #ondsp.fp<format = f32, contract = fast>
  } : (tensor<4xf32>, tensor<3xf32>, tensor<2xf32>)
      -> (tensor<4xf32>, tensor<2xf32>)
  return %output, %next : tensor<4xf32>, tensor<2xf32>
}
