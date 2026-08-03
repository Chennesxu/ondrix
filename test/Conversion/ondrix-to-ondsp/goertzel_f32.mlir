// RUN: ondrix-opt %s --convert-ondrix-to-ondsp | FileCheck %s --implicit-check-not=fastmath

// The fast contract admits both the fused and the separate form of the one
// multiply-add. The lowering SELECTS the fused member and emits that choice
// unflagged: carrying the declaration onward would hand the choice to the
// backend, which de-fuses a reassoc-flagged fma on the pinned toolchain.
// CHECK-LABEL: func.func @f32_goertzel_fast
// CHECK: %[[C2:.*]] = arith.constant
// CHECK: scf.for {{.*}} iter_args
// CHECK: math.fma %[[C2]], %{{[^ ]*}}, %{{[^ ]*}} : f32
// CHECK: arith.subf
// CHECK-NOT: fastmath
// CHECK: arith.mulf %[[C2]]
// CHECK: arith.addf
// CHECK: arith.subf
// CHECK-NOT: ondsp.round_shift
// CHECK-NOT: ondsp.sat_cast
func.func @f32_goertzel_fast(%input: tensor<16xf32>) -> tensor<1xf32> {
  %energy = ondrix.goertzel %input {
    bin = 3,
    numeric = #ondsp.fp<format = f32, contract = fast>
  } : (tensor<16xf32>) -> tensor<1xf32>
  return %energy : tensor<1xf32>
}

// The only gate on the quarter-turn constant: the energy it feeds is
// bit-identical either way, so no object test can distinguish it.
// CHECK-LABEL: func.func @f32_goertzel_quarter_turn
// CHECK: arith.constant 0.000000e+00 : f32
func.func @f32_goertzel_quarter_turn(%input: tensor<16xf32>) -> tensor<1xf32> {
  %energy = ondrix.goertzel %input {
    bin = 4,
    numeric = #ondsp.fp<format = f32, contract = off>
  } : (tensor<16xf32>) -> tensor<1xf32>
  return %energy : tensor<1xf32>
}
