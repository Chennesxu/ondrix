// RUN: ondrix-opt %s --convert-ondrix-to-ondsp="fft-loops" | FileCheck %s --implicit-check-not=ondsp.

// The peeled unit and quarter-turn phases ARE the zero slots of the table, so
// losing a peel reads zero and collapses the stage instead of slowing it.
// CHECK-LABEL: func.func @f32_cfft16_loops
// CHECK: arith.constant dense<[0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.707106769, -0.707106769, 0.000000e+00, 0.000000e+00, -0.707106769, -0.707106769, 0.000000e+00, 0.000000e+00, 0.923879504, -0.382683426, 0.707106769, -0.707106769, 0.382683426, -0.923879504, 0.000000e+00, 0.000000e+00, -0.382683426, -0.923879504, -0.707106769, -0.707106769, -0.923879504, -0.382683426]> : tensor<32xf32>
// CHECK: arith.constant dense<[0, 8, 4, 12, 2, 10, 6, 14, 1, 9, 5, 13, 3, 11, 7, 15]> : tensor<16xi64>

// Nine loops: gather, the peeled half = 1 stage, the stage loop, and inside it
// two peeled-phase loops and two group/phase pairs.
// CHECK-COUNT-9: scf.for
// CHECK-NOT: scf.for
func.func @f32_cfft16_loops(%input: tensor<32xf32>) -> tensor<32xf32> {
  %result = ondrix.cfft %input {
    direction = #ondrix.cfft_direction<forward>,
    layout = #ondsp.cx_layout<interleaved>,
    numeric = #ondsp.fp<format = f32, contract = off>
  } : (tensor<32xf32>) -> tensor<32xf32>
  return %result : tensor<32xf32>
}

// Eleven: the same nine plus the Hermitian mirror and the scaled real export.
// CHECK-LABEL: func.func @f32_irfft16_loops
// CHECK-COUNT-11: scf.for
// CHECK-NOT: scf.for
func.func @f32_irfft16_loops(%input: tensor<18xf32>) -> tensor<16xf32> {
  %result = ondrix.irfft %input {
    layout = #ondsp.cx_layout<interleaved>,
    numeric = #ondsp.fp<format = f32, contract = off>
  } : (tensor<18xf32>) -> tensor<16xf32>
  return %result : tensor<16xf32>
}
