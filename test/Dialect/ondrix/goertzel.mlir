// RUN: ondrix-opt %s | ondrix-opt | FileCheck %s

// CHECK-LABEL: func.func @goertzel64_5
// CHECK: ondrix.goertzel
// CHECK-SAME: bin = 5
// CHECK-SAME: frac = 15
func.func @goertzel64_5(%input: tensor<64xi16>) -> tensor<1xi64> {
  %energy = ondrix.goertzel %input {
    bin = 5 : i64,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<64xi16>) -> tensor<1xi64>
  return %energy : tensor<1xi64>
}

// Non-power-of-two lengths are legal: the recursion has no radix
// structure, only the per-bin coefficient.
// CHECK-LABEL: func.func @goertzel100_13
// CHECK: ondrix.goertzel
// CHECK-SAME: bin = 13
func.func @goertzel100_13(%input: tensor<100xi16>) -> tensor<1xi64> {
  %energy = ondrix.goertzel %input {
    bin = 13 : i64,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<100xi16>) -> tensor<1xi64>
  return %energy : tensor<1xi64>
}
