// RUN: ondrix-opt %s --convert-ondrix-to-ondsp | FileCheck %s

// One sample loop carrying the two i16 states as scalar iter_args; per
// step one round_shift product boundary and one saturating combine, then
// the final product boundary and the exact i64 energy expression. The
// doubled coefficient constant 2 * q15(cos(2*pi*5/64)) = 2 * 28899 is
// pinned (independently derived with 50-digit mpmath).

// CHECK-LABEL: func.func @goertzel64_5
// CHECK: arith.constant 57798 : i64
// CHECK: scf.for
// CHECK: ondsp.round_shift
// CHECK: ondsp.sat_cast
// CHECK: ondsp.round_shift
// CHECK: arith.muli
// CHECK-NOT: ondrix.goertzel
func.func @goertzel64_5(%input: tensor<64xi16>) -> tensor<1xi64> {
  %energy = ondrix.goertzel %input {
    bin = 5 : i64,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<64xi16>) -> tensor<1xi64>
  return %energy : tensor<1xi64>
}
