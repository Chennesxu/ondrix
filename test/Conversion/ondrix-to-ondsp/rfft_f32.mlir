// RUN: ondrix-opt %s --convert-ondrix-to-ondsp | FileCheck %s --implicit-check-not=ondsp.

// The real profiles share the complex recursion, so what is checked here is
// the embedding, the canonical zeros, and the exact-sign-flip conjugation.

// Forty products, the same count as the size-16 complex transform: embedding a
// real input adds no arithmetic.
// CHECK-LABEL: func.func @f32_rfft16
// CHECK-DAG: arith.constant 0.000000e+00 : f32
// CHECK-COUNT-40: arith.mulf
// CHECK-NOT: arith.mulf
func.func @f32_rfft16(%input: tensor<16xf32>) -> tensor<18xf32> {
  %result = ondrix.rfft %input {
    layout = #ondsp.cx_layout<interleaved>,
    numeric = #ondsp.fp<format = f32, contract = off>
  } : (tensor<16xf32>) -> tensor<18xf32>
  return %result : tensor<18xf32>
}

// Twenty-four negations are seventeen from the complex recursion plus the
// seven exact conjugations; the packed profiles need a saturating select.
// CHECK-LABEL: func.func @f32_irfft16
// CHECK-COUNT-24: arith.negf
// CHECK-NOT: arith.negf
// CHECK-NOT: arith.select
func.func @f32_irfft16(%input: tensor<18xf32>) -> tensor<16xf32> {
  %result = ondrix.irfft %input {
    layout = #ondsp.cx_layout<interleaved>,
    numeric = #ondsp.fp<format = f32, contract = off>
  } : (tensor<18xf32>) -> tensor<16xf32>
  return %result : tensor<16xf32>
}
