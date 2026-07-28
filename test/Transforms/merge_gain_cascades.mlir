// RUN: ondrix-opt %s --merge-ondrix-gain-cascades | FileCheck %s

// The merge is authorized per cascade by an exhaustive compile-time
// certificate over all 65536 i16 inputs; everything the certificate does
// not prove stays untouched (the unmerged program is the safe state).

// Halve then negate-halve: certified, merges to one -8192 gain carrying
// the provenance of the proof.
// CHECK-LABEL: func.func @certified_merge
// CHECK: = ondrix.gain %
// CHECK-SAME: gain = -8192
// CHECK-SAME: exhaustive_inputs = 65536
// CHECK-SAME: inner_gain = 16384
// CHECK-SAME: outer_gain = -16384
// CHECK-NOT: = ondrix.gain %
func.func @certified_merge(%input: tensor<8xi16>) -> tensor<8xi16> {
  %halved = ondrix.gain %input {
    gain = 16384 : i64,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<8xi16>) -> tensor<8xi16>
  %negated = ondrix.gain %halved {
    gain = -16384 : i64,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<8xi16>) -> tensor<8xi16>
  return %negated : tensor<8xi16>
}

// A small gain followed by a near-unity gain: the second requantization
// provably never moves any reachable value, so the cascade collapses to
// the inner constant alone. Certified, not assumed.
// CHECK-LABEL: func.func @near_unity_absorption
// CHECK: = ondrix.gain %
// CHECK-SAME: gain = 3125
// CHECK-SAME: inner_gain = 3125
// CHECK-SAME: outer_gain = 32764
// CHECK-NOT: = ondrix.gain %
func.func @near_unity_absorption(%input: tensor<8xi16>) -> tensor<8xi16> {
  %scaled = ondrix.gain %input {
    gain = 3125 : i64,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<8xi16>) -> tensor<8xi16>
  %adjusted = ondrix.gain %scaled {
    gain = 32764 : i64,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<8xi16>) -> tensor<8xi16>
  return %adjusted : tensor<8xi16>
}

// The A3 family-5 witness pair: the cascade and the quantized-product
// merge differ on 10038 of 65536 inputs, so the certificate rejects the
// rewrite and both operations survive.
// CHECK-LABEL: func.func @witness_not_merged
// CHECK: = ondrix.gain %
// CHECK-SAME: gain = 22938
// CHECK: = ondrix.gain %
// CHECK-SAME: gain = 19661
func.func @witness_not_merged(%input: tensor<8xi16>) -> tensor<8xi16> {
  %first = ondrix.gain %input {
    gain = 22938 : i64,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<8xi16>) -> tensor<8xi16>
  %second = ondrix.gain %first {
    gain = 19661 : i64,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<8xi16>) -> tensor<8xi16>
  return %second : tensor<8xi16>
}

// A multi-use intermediate is observable and never merged away, even for
// a pair whose certificate would pass.
// CHECK-LABEL: func.func @multi_use_not_merged
// CHECK: = ondrix.gain %
// CHECK-SAME: gain = 16384
// CHECK: = ondrix.gain %
// CHECK-SAME: gain = -16384
func.func @multi_use_not_merged(%input: tensor<8xi16>) -> (tensor<8xi16>, tensor<8xi16>) {
  %halved = ondrix.gain %input {
    gain = 16384 : i64,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<8xi16>) -> tensor<8xi16>
  %negated = ondrix.gain %halved {
    gain = -16384 : i64,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<8xi16>) -> tensor<8xi16>
  return %negated, %halved : tensor<8xi16>, tensor<8xi16>
}

// Fixpoint with fail-closed tail: three halvings merge the inner pair
// (certified 8192) but the certificate rejects folding the third stage
// (8192 then 16384 is NOT 4096 — 8192 inputs diverge), so exactly two
// operations remain.
// CHECK-LABEL: func.func @triple_halving_partial
// CHECK: = ondrix.gain %
// CHECK-SAME: gain = 8192
// CHECK: = ondrix.gain %
// CHECK-SAME: gain = 16384
// CHECK-NOT: = ondrix.gain %
func.func @triple_halving_partial(%input: tensor<8xi16>) -> tensor<8xi16> {
  %first = ondrix.gain %input {
    gain = 16384 : i64,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<8xi16>) -> tensor<8xi16>
  %second = ondrix.gain %first {
    gain = 16384 : i64,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<8xi16>) -> tensor<8xi16>
  %third = ondrix.gain %second {
    gain = 16384 : i64,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<8xi16>) -> tensor<8xi16>
  return %third : tensor<8xi16>
}
