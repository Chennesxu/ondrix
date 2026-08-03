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

// The row product is the declared per-term event and nothing follows the sum.
//
// The constants are pinned because they are not derived from a specification
// a reader could check: they are the binary32 narrowing of a binary64 cosine
// evaluated by the build's libm. That value moves only if the evaluation
// errs by about 2^17 binary64 ulps, which no usable libm does, but C requires
// no accuracy of `cos` at all — so the dependency is pinned rather than
// argued away. Extent 8, whose eight distinct magnitudes are the full table:
// CHECK-LABEL: func.func @f32_dct
// CHECK-DAG: arith.constant 0.195090324 : f32
// CHECK-DAG: arith.constant 0.382683426 : f32
// CHECK-DAG: arith.constant 0.555570245 : f32
// CHECK-DAG: arith.constant 0.707106769 : f32
// CHECK-DAG: arith.constant 0.831469595 : f32
// CHECK-DAG: arith.constant 0.923879504 : f32
// CHECK-DAG: arith.constant 0.98078525 : f32
// CHECK-DAG: arith.constant 1.000000e+00 : f32
// CHECK: math.fma
// CHECK-NOT: ondsp.round_shift
func.func @f32_dct(%input: tensor<8xf32>) -> tensor<8xf32> {
  %result = ondrix.dct %input {
    input_numeric = #ondsp.fp<format = f32, contract = fma>,
    output_numeric = #ondsp.fp<format = f32, contract = fma>
  } : (tensor<8xf32>) -> tensor<8xf32>
  return %result : tensor<8xf32>
}
