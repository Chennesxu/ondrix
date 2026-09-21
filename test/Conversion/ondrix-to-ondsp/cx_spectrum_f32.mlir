// RUN: ondrix-opt %s --convert-ondrix-to-ondsp | FileCheck %s

// The bin index is doubled to reach the real element, so one loop trip reads
// two adjacent elements and writes one.
func.func @power_off(%input: tensor<8xf32>) -> tensor<4xf32> {
  %out = ondrix.cx_power %input {
    layout = #ondsp.cx_layout<interleaved>,
    numeric = #ondsp.fp<format = f32, contract = off>
  } : (tensor<8xf32>) -> tensor<4xf32>
  return %out : tensor<4xf32>
}

// CHECK-LABEL: func.func @power_off
// CHECK: scf.for
// CHECK: %[[REAL_INDEX:.*]] = arith.muli
// CHECK: %[[IMAG_INDEX:.*]] = arith.addi %[[REAL_INDEX]]
// CHECK: %[[REAL:.*]] = tensor.extract %{{.*}}[%[[REAL_INDEX]]]
// CHECK: %[[IMAG:.*]] = tensor.extract %{{.*}}[%[[IMAG_INDEX]]]
// CHECK: %[[SEED:.*]] = arith.mulf %[[REAL]], %[[REAL]]
// CHECK: %[[SQUARE:.*]] = arith.mulf %[[IMAG]], %[[IMAG]]
// CHECK: arith.addf %[[SEED]], %[[SQUARE]]
// CHECK-NOT: math.fma
// CHECK: tensor.insert

// The sum seeds on the first product, so the fused contract spends one
// multiply and one fma rather than two multiplies and an add.
func.func @power_fma(%input: tensor<8xf32>) -> tensor<4xf32> {
  %out = ondrix.cx_power %input {
    layout = #ondsp.cx_layout<interleaved>,
    numeric = #ondsp.fp<format = f32, contract = fma>
  } : (tensor<8xf32>) -> tensor<4xf32>
  return %out : tensor<4xf32>
}

// CHECK-LABEL: func.func @power_fma
// CHECK: %[[SEED:.*]] = arith.mulf
// CHECK: math.fma %{{.*}}, %{{.*}}, %[[SEED]]
// CHECK-NOT: arith.addf

func.func @magnitude_off(%input: tensor<66xf32>) -> tensor<33xf32> {
  %out = ondrix.cx_magnitude %input {
    layout = #ondsp.cx_layout<interleaved>,
    numeric = #ondsp.fp<format = f32, contract = off>
  } : (tensor<66xf32>) -> tensor<33xf32>
  return %out : tensor<33xf32>
}

// CHECK-LABEL: func.func @magnitude_off
// CHECK: %[[SUM:.*]] = arith.addf
// CHECK: math.sqrt %[[SUM]]
// CHECK: tensor.insert

// The magnitude is the root of exactly the power sum, so the fused contract
// reaches the same fma before it.
func.func @magnitude_fast(%input: tensor<8xf32>) -> tensor<4xf32> {
  %out = ondrix.cx_magnitude %input {
    layout = #ondsp.cx_layout<interleaved>,
    numeric = #ondsp.fp<format = f32, contract = fast>
  } : (tensor<8xf32>) -> tensor<4xf32>
  return %out : tensor<4xf32>
}

// CHECK-LABEL: func.func @magnitude_fast
// CHECK: %[[SUM:.*]] = math.fma
// CHECK: math.sqrt %[[SUM]]
