// RUN: ondrix-opt %s --relax-ondsp-unreachable-saturation --split-input-file | FileCheck %s

!acc = !ondsp.acc<storage = i64, frac = 30, signed, update_overflow = wrap>

// Four full-scale Q15 products reach 2^32, so a reading at frac 11 needs
// fourteen bits and the declared clamp cannot fire.
// CHECK-LABEL: func.func @headroom_relaxes(
// CHECK: ondsp.round_shift
// CHECK-SAME: post_shift_right = 19, rounding = nearest_even, overflow = wrap
func.func @headroom_relaxes(%x: memref<4xi16>) -> i16 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %k = arith.constant 32767 : i32
  %z = ondsp.acc_zero : !acc
  %a = memref.load %x[%c0] : memref<4xi16>
  %ae = arith.extsi %a : i16 to i32
  %ap = arith.muli %ae, %k : i32
  %s0 = ondsp.acc_add_term %z, %ap {term_numeric = #ondsp.fixed<signed, storage = i32, frac = 30>} : (!acc, i32) -> !acc
  %b = memref.load %x[%c1] : memref<4xi16>
  %be = arith.extsi %b : i16 to i32
  %bp = arith.muli %be, %k : i32
  %s1 = ondsp.acc_add_term %s0, %bp {term_numeric = #ondsp.fixed<signed, storage = i32, frac = 30>} : (!acc, i32) -> !acc
  %e = ondsp.acc_export %s1 {dst = #ondsp.fixed<signed, storage = i64, frac = 30>, overflow = #ondsp.overflow<saturate>, rounding = #ondsp.rounding<nearest_even>} : (!acc) -> i64
  %r = ondsp.round_shift %e {scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 19, rounding = nearest_even, overflow = saturate, saturate_to = i16>} : (i64) -> i16
  return %r : i16
}

// -----

!acc = !ondsp.acc<storage = i64, frac = 30, signed, update_overflow = wrap>

// The same chain read at frac 15 reaches 2^16, past the i16 rail: the only
// difference from the case above is the shift, and the clamp survives.
// CHECK-LABEL: func.func @rail_reachable_keeps_clamp(
// CHECK: ondsp.round_shift
// CHECK-SAME: post_shift_right = 15, rounding = nearest_even, overflow = saturate
func.func @rail_reachable_keeps_clamp(%x: memref<4xi16>) -> i16 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %k = arith.constant 32767 : i32
  %z = ondsp.acc_zero : !acc
  %a = memref.load %x[%c0] : memref<4xi16>
  %ae = arith.extsi %a : i16 to i32
  %ap = arith.muli %ae, %k : i32
  %s0 = ondsp.acc_add_term %z, %ap {term_numeric = #ondsp.fixed<signed, storage = i32, frac = 30>} : (!acc, i32) -> !acc
  %b = memref.load %x[%c1] : memref<4xi16>
  %be = arith.extsi %b : i16 to i32
  %bp = arith.muli %be, %k : i32
  %s1 = ondsp.acc_add_term %s0, %bp {term_numeric = #ondsp.fixed<signed, storage = i32, frac = 30>} : (!acc, i32) -> !acc
  %e = ondsp.acc_export %s1 {dst = #ondsp.fixed<signed, storage = i64, frac = 30>, overflow = #ondsp.overflow<saturate>, rounding = #ondsp.rounding<nearest_even>} : (!acc) -> i64
  %r = ondsp.round_shift %e {scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 15, rounding = nearest_even, overflow = saturate, saturate_to = i16>} : (i64) -> i16
  return %r : i16
}

// -----

!acc = !ondsp.acc<storage = i64, frac = 30, signed, update_overflow = wrap>

// Sixteen trips of one full-scale term reach 2^34; read at frac 9 that is
// fourteen bits, so the counted loop earns the same relaxation.
// CHECK-LABEL: func.func @counted_loop_relaxes(
// CHECK: ondsp.round_shift
// CHECK-SAME: post_shift_right = 21, rounding = nearest_even, overflow = wrap
func.func @counted_loop_relaxes(%x: memref<32xi16>) -> i16 {
  %c0 = arith.constant 0 : index
  %c2 = arith.constant 2 : index
  %c32 = arith.constant 32 : index
  %k = arith.constant 32767 : i32
  %z = ondsp.acc_zero : !acc
  %acc = scf.for %i = %c0 to %c32 step %c2 iter_args(%it = %z) -> (!acc) {
    %v = memref.load %x[%i] : memref<32xi16>
    %ve = arith.extsi %v : i16 to i32
    %p = arith.muli %ve, %k : i32
    %n = ondsp.acc_add_term %it, %p {term_numeric = #ondsp.fixed<signed, storage = i32, frac = 30>} : (!acc, i32) -> !acc
    scf.yield %n : !acc
  }
  %e = ondsp.acc_export %acc {dst = #ondsp.fixed<signed, storage = i64, frac = 30>, overflow = #ondsp.overflow<saturate>, rounding = #ondsp.rounding<nearest_even>} : (!acc) -> i64
  %r = ondsp.round_shift %e {scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 21, rounding = nearest_even, overflow = saturate, saturate_to = i16>} : (i64) -> i16
  return %r : i16
}

// -----

!acc = !ondsp.acc<storage = i64, frac = 30, signed, update_overflow = wrap>

// A trip count the pass cannot read bounds nothing; the same loop body at the
// same shift as the case above keeps its clamp.
// CHECK-LABEL: func.func @dynamic_trip_count_keeps_clamp(
// CHECK: ondsp.round_shift
// CHECK-SAME: post_shift_right = 21, rounding = nearest_even, overflow = saturate
func.func @dynamic_trip_count_keeps_clamp(%x: memref<32xi16>, %n: index) -> i16 {
  %c0 = arith.constant 0 : index
  %c2 = arith.constant 2 : index
  %k = arith.constant 32767 : i32
  %z = ondsp.acc_zero : !acc
  %acc = scf.for %i = %c0 to %n step %c2 iter_args(%it = %z) -> (!acc) {
    %v = memref.load %x[%i] : memref<32xi16>
    %ve = arith.extsi %v : i16 to i32
    %p = arith.muli %ve, %k : i32
    %m = ondsp.acc_add_term %it, %p {term_numeric = #ondsp.fixed<signed, storage = i32, frac = 30>} : (!acc, i32) -> !acc
    scf.yield %m : !acc
  }
  %e = ondsp.acc_export %acc {dst = #ondsp.fixed<signed, storage = i64, frac = 30>, overflow = #ondsp.overflow<saturate>, rounding = #ondsp.rounding<nearest_even>} : (!acc) -> i64
  %r = ondsp.round_shift %e {scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 21, rounding = nearest_even, overflow = saturate, saturate_to = i16>} : (i64) -> i16
  return %r : i16
}

// -----

// A value with no provenance carries only its type's range, and 2^63 read at
// frac 11 is far past the rail.
// CHECK-LABEL: func.func @opaque_input_keeps_clamp(
// CHECK: overflow = saturate
func.func @opaque_input_keeps_clamp(%v: i64) -> i16 {
  %r = ondsp.round_shift %v {scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 19, rounding = nearest_even, overflow = saturate, saturate_to = i16>} : (i64) -> i16
  return %r : i16
}
