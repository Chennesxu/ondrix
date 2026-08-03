// RUN: ondrix-opt %s --convert-ondrix-to-ondsp | FileCheck %s
// RUN: ondrix-opt %s --convert-ondrix-to-ondsp="sliding-window-reuse=true" | FileCheck %s --check-prefix=REUSE

// The floating-point average declines the incremental form: the reuse is
// value-neutral only for exact integer window sums, so both runs emit the
// same recomputed windows.
// CHECK-LABEL: func.func @f32_moving_average
// CHECK-COUNT-6: arith.divf
// CHECK-NOT: arith.divf
// REUSE-LABEL: func.func @f32_moving_average
// REUSE-COUNT-6: arith.divf
// REUSE-NOT: arith.divf
func.func @f32_moving_average(%input: tensor<8xf32>) -> tensor<6xf32> {
  %result = ondrix.moving_average %input {
    window = 3 : i64,
    numeric = #ondsp.fp<format = f32, contract = off>
  } : (tensor<8xf32>) -> tensor<6xf32>
  return %result : tensor<6xf32>
}

// The f32 coefficient table is the correctly rounded cosine; the row product
// is the declared per-term event and nothing follows the sum.
// CHECK-LABEL: func.func @f32_dct
// CHECK: math.fma
// CHECK-NOT: ondsp.round_shift
func.func @f32_dct(%input: tensor<8xf32>) -> tensor<8xf32> {
  %result = ondrix.dct %input {
    input_numeric = #ondsp.fp<format = f32, contract = fma>,
    output_numeric = #ondsp.fp<format = f32, contract = fma>
  } : (tensor<8xf32>) -> tensor<8xf32>
  return %result : tensor<8xf32>
}
