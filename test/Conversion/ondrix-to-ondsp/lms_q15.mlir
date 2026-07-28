// RUN: ondrix-opt %s --convert-ondrix-to-ondsp | FileCheck %s

// The whole recursion is loop-form: an outer sample loop carries the
// quantized weight state and the error tensor as iter_args, an inner
// exact-i64 accumulation loop feeds one output round_shift, the error is
// one saturating cast, the step is one round_shift, and the update loop
// takes one round_shift plus one saturating cast per tap. Every rounding
// boundary of the contract appears exactly once per position.

// CHECK-LABEL: func.func @lms4_q15
// CHECK: scf.for
// CHECK: scf.for
// CHECK: arith.muli
// CHECK: ondsp.round_shift
// CHECK: arith.subi
// CHECK: ondsp.sat_cast
// CHECK: tensor.insert
// CHECK: ondsp.round_shift
// CHECK: scf.for
// CHECK: ondsp.round_shift
// CHECK: ondsp.sat_cast
// CHECK-NOT: ondrix.lms
func.func @lms4_q15(%x: tensor<16xi16>, %d: tensor<16xi16>, %w: tensor<4xi16>)
    -> (tensor<16xi16>, tensor<4xi16>) {
  %e, %wf = ondrix.lms %x, %d, %w {
    step_size = 8192 : i64,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<16xi16>, tensor<16xi16>, tensor<4xi16>) -> (tensor<16xi16>, tensor<4xi16>)
  return %e, %wf : tensor<16xi16>, tensor<4xi16>
}
