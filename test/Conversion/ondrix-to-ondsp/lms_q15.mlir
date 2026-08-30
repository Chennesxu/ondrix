// RUN: ondrix-opt %s --convert-ondrix-to-ondsp | FileCheck %s

// The whole recursion is loop-form: the outer sample loop pair carries the
// quantized weight state and the error tensor as iter_args, an inner
// exact-i64 accumulation loop feeds one output round_shift, the error is
// one saturating cast, the step is one round_shift, and the update loop
// takes one round_shift plus one saturating cast per tap. Every rounding
// boundary of the contract appears exactly once per position.

// CHECK-LABEL: func.func @lms4_q15
// CHECK-DAG: %[[N:.*]] = arith.constant 16 : index
// CHECK-DAG: %[[PEEL:.*]] = arith.constant 3 : index
// CHECK: %[[PRE:.*]]:2 = scf.for %{{.*}} to %[[PEEL]] step
// CHECK: arith.cmpi sge
// CHECK: arith.maxsi
// CHECK: arith.select
// CHECK: arith.muli
// CHECK: ondsp.round_shift
// CHECK: arith.subi
// CHECK: ondsp.sat_cast
// CHECK: tensor.insert
// CHECK: ondsp.round_shift
// CHECK: scf.for
// CHECK: ondsp.round_shift
// CHECK: ondsp.sat_cast
// Past sample K - 1 every tap offset n - k is nonnegative by construction.
// CHECK: %[[MAIN:.*]]:2 = scf.for %{{.*}} = %[[PEEL]] to %[[N]] step %{{.*}} iter_args(%{{.*}} = %[[PRE]]#0, %{{.*}} = %[[PRE]]#1)
// CHECK-NOT: arith.cmpi
// CHECK-NOT: arith.maxsi
// CHECK-NOT: arith.select
// CHECK: return %[[MAIN]]#1, %[[MAIN]]#0
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

// A single tap has no prehistory: the one sample loop still takes the fresh
// error tensor, so the guarded region is absent rather than emitted zero-trip.
// CHECK-LABEL: func.func @lms1_q15
// CHECK-DAG: %[[N:.*]] = arith.constant 32 : index
// CHECK: %[[ERRORS:.*]] = tensor.empty() : tensor<32xi16>
// CHECK-NOT: arith.cmpi
// CHECK: %[[ONLY:.*]]:2 = scf.for %{{.*}} to %[[N]] step %{{.*}} iter_args(%{{.*}} = %{{.*}}, %{{.*}} = %[[ERRORS]])
// CHECK-NOT: arith.cmpi
// CHECK-NOT: arith.maxsi
// CHECK-NOT: arith.select
// CHECK: return %[[ONLY]]#1, %[[ONLY]]#0
func.func @lms1_q15(%x: tensor<32xi16>, %d: tensor<32xi16>, %w: tensor<1xi16>)
    -> (tensor<32xi16>, tensor<1xi16>) {
  %e, %wf = ondrix.lms %x, %d, %w {
    step_size = 8192 : i64,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<32xi16>, tensor<32xi16>, tensor<1xi16>) -> (tensor<32xi16>, tensor<1xi16>)
  return %e, %wf : tensor<32xi16>, tensor<1xi16>
}

// N at most K - 1 keeps every sample inside the prehistory, so the unguarded
// region is the empty one.
// CHECK-LABEL: func.func @lms8_short_q15
// CHECK-DAG: %[[N:.*]] = arith.constant 4 : index
// CHECK: %[[ONLY:.*]]:2 = scf.for %{{.*}} to %[[N]] step
// CHECK: arith.cmpi sge
// CHECK: arith.maxsi
// CHECK: arith.select
// CHECK: return %[[ONLY]]#1, %[[ONLY]]#0
func.func @lms8_short_q15(%x: tensor<4xi16>, %d: tensor<4xi16>, %w: tensor<8xi16>)
    -> (tensor<4xi16>, tensor<8xi16>) {
  %e, %wf = ondrix.lms %x, %d, %w {
    step_size = 8192 : i64,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<4xi16>, tensor<4xi16>, tensor<8xi16>) -> (tensor<4xi16>, tensor<8xi16>)
  return %e, %wf : tensor<4xi16>, tensor<8xi16>
}
