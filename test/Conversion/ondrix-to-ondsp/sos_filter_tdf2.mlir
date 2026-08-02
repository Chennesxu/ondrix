// RUN: ondrix-opt %s --convert-ondrix-to-ondsp | FileCheck %s

// CHECK-LABEL: func.func @dynamic_fma
// CHECK-COUNT-2: cf.assert
// CHECK: tensor.empty(%{{.*}}) : tensor<?xf32>
// CHECK: %[[SAMPLES:.*]]:2 = scf.for
// CHECK: tensor.extract
// CHECK: %[[SECTIONS:.*]]:2 = scf.for
// CHECK-COUNT-8: tensor.extract
// CHECK: arith.mulf
// CHECK: math.fma
// CHECK: arith.mulf
// CHECK: math.fma
// CHECK: arith.addf
// CHECK: arith.mulf
// CHECK: math.fma
// CHECK-COUNT-2: tensor.insert
// CHECK: scf.yield
// CHECK: tensor.insert
// CHECK: return %[[SAMPLES]]#0, %[[SAMPLES]]#1
// CHECK-NOT: ondrix.sos_filter_tdf2
func.func @dynamic_fma(
    %input: tensor<?xf32>, %coeffs: tensor<?x5xf32>,
    %scales: tensor<?xf32>, %state: tensor<?x2xf32>)
    -> (tensor<?xf32>, tensor<?x2xf32>) {
  %output, %next = ondrix.sos_filter_tdf2 %input, %coeffs, %scales, %state {
    numeric = #ondsp.fp<format = f32, contract = fma>
  } : (tensor<?xf32>, tensor<?x5xf32>, tensor<?xf32>, tensor<?x2xf32>)
      -> (tensor<?xf32>, tensor<?x2xf32>)
  return %output, %next : tensor<?xf32>, tensor<?x2xf32>
}

// CHECK-LABEL: func.func @static_off
// CHECK-NOT: math.fma
// CHECK: arith.mulf
// CHECK: arith.mulf
// CHECK: arith.addf
// CHECK: arith.mulf
// CHECK: arith.mulf
// CHECK: arith.addf
// CHECK: arith.addf
// CHECK: arith.mulf
// CHECK: arith.mulf
// CHECK: arith.addf
// CHECK-NOT: ondrix.sos_filter_tdf2
func.func @static_off(
    %input: tensor<4xf32>, %coeffs: tensor<2x5xf32>,
    %scales: tensor<2xf32>, %state: tensor<2x2xf32>)
    -> (tensor<4xf32>, tensor<2x2xf32>) {
  %output, %next = ondrix.sos_filter_tdf2 %input, %coeffs, %scales, %state {
    numeric = #ondsp.fp<format = f32, contract = off>
  } : (tensor<4xf32>, tensor<2x5xf32>, tensor<2xf32>, tensor<2x2xf32>)
      -> (tensor<4xf32>, tensor<2x2xf32>)
  return %output, %next : tensor<4xf32>, tensor<2x2xf32>
}

// CHECK-LABEL: func.func @static_fast
// CHECK: arith.mulf {{.*}} fastmath<reassoc,contract>
// CHECK: math.fma {{.*}} fastmath<reassoc,contract>
// CHECK: arith.addf {{.*}} fastmath<reassoc,contract>
// CHECK-NOT: ondrix.sos_filter_tdf2
func.func @static_fast(
    %input: tensor<4xf32>, %coeffs: tensor<2x5xf32>,
    %scales: tensor<2xf32>, %state: tensor<2x2xf32>)
    -> (tensor<4xf32>, tensor<2x2xf32>) {
  %output, %next = ondrix.sos_filter_tdf2 %input, %coeffs, %scales, %state {
    numeric = #ondsp.fp<format = f32, contract = fast>
  } : (tensor<4xf32>, tensor<2x5xf32>, tensor<2xf32>, tensor<2x2xf32>)
      -> (tensor<4xf32>, tensor<2x2xf32>)
  return %output, %next : tensor<4xf32>, tensor<2x2xf32>
}
