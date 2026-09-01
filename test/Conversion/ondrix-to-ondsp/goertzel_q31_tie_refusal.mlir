// RUN: not ondrix-opt %s --convert-ondrix-to-ondsp 2>&1 | FileCheck %s

// The Q31 coefficient guard is not decoration the way the Q15 one is. Measured
// over all 4198399 admissible (k, N) sites, 1017 sit closer to a rounding tie
// than the 2^-13 LSB guard, so a binary64 evaluation cannot prove they round
// like the real-valued definition. Bin 612 at N = 2093 is one of them and
// fails closed; bin 613 at the SAME extent is admissible, so this pins a
// per-bin decision rather than an extent-wide one.

// CHECK: failed to legalize operation 'ondrix.goertzel'
func.func @goertzel_q31_inadmissible_bin(%input: tensor<2093xi32>) -> tensor<1xi64> {
  %energy = ondrix.goertzel %input {
    bin = 612 : i64,
    numeric = #ondsp.fixed<signed, storage = i32, frac = 31>,
    state_rounding = #ondsp.rounding<nearest_even>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<2093xi32>) -> tensor<1xi64>
  return %energy : tensor<1xi64>
}
