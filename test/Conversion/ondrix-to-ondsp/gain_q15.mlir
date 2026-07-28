// RUN: ondrix-opt %s --convert-ondrix-to-ondsp | FileCheck %s

// Each element takes one exact i64 product against the constant and one
// nearest-even saturating ondsp.round_shift by 15 back to Q1.15 — the
// single declared boundary of the contract — inside one elementwise loop
// (the operation admits extents up to 4096, where unrolled insert chains
// are quadratic to bufferize).

// CHECK-LABEL: func.func @gain4_q15
// CHECK: arith.constant 19661 : i64
// CHECK: scf.for
// CHECK: arith.extsi
// CHECK: arith.muli
// CHECK: ondsp.round_shift
// CHECK-NOT: ondsp.round_shift
// CHECK: tensor.insert
// CHECK-NOT: ondrix.gain
func.func @gain4_q15(%input: tensor<4xi16>) -> tensor<4xi16> {
  %result = ondrix.gain %input {
    gain = 19661 : i64,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<4xi16>) -> tensor<4xi16>
  return %result : tensor<4xi16>
}
