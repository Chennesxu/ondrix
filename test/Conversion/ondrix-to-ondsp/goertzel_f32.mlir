// RUN: ondrix-opt %s --convert-ondrix-to-ondsp | FileCheck %s

// Only the coefficient product and its input addition form a multiply-add,
// so that is the only site the contract indexes. The state subtraction and
// the closing energy expression stay unflagged in every mode, which bounds
// what fast may vary to that one site.
// CHECK-LABEL: func.func @f32_goertzel_fast
// CHECK: %[[C2:.*]] = arith.constant
// CHECK: scf.for {{.*}} iter_args
// CHECK: math.fma %[[C2]], {{.*}} fastmath<reassoc,contract> : f32
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

// binary64 cannot represent pi/2, so an unsnapped libm cosine would give
// about 1e-16 here instead of the zero the quarter-turn bin names.
// CHECK-LABEL: func.func @f32_goertzel_quarter_turn
// CHECK: arith.constant 0.000000e+00 : f32
func.func @f32_goertzel_quarter_turn(%input: tensor<16xf32>) -> tensor<1xf32> {
  %energy = ondrix.goertzel %input {
    bin = 4,
    numeric = #ondsp.fp<format = f32, contract = off>
  } : (tensor<16xf32>) -> tensor<1xf32>
  return %energy : tensor<1xf32>
}
