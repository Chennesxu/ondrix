// RUN: not ondrix-opt %s --convert-ondrix-to-ondsp 2>&1 | FileCheck %s

// The one inadmissible coefficient in the whole legal domain: a binary64
// sweep of every (N in [2, 4096], bin in [0, N/2]) pair finds exactly one
// cosine inside the 2^-20 quantization tie guard, at N = 3289, bin = 610.
// The verifier accepts the operation (the bound is a compile-time
// numerical fact, not a shape fact) and the lowering must fail closed
// instead of quantizing through the tie.

// CHECK: failed to legalize operation 'ondrix.goertzel'

func.func @goertzel_tie_guard(%input: tensor<3289xi16>) -> tensor<1xi64> {
  %energy = ondrix.goertzel %input {
    bin = 610 : i64,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<3289xi16>) -> tensor<1xi64>
  return %energy : tensor<1xi64>
}
