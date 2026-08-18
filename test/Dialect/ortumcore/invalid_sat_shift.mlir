// RUN: ondrix-opt %s -split-input-file -verify-diagnostics

// The scaling immediate of the saturating add/sub is narrower than the
// accumulator readout's, so it needs its own bound rather than the readout's.
func.func @scaled_add_shift_above_range(%lhs: i32, %rhs: i32) -> i32 {
  // expected-error @+1 {{scaled saturating add/sub shift must lie in [0, 3]}}
  %0 = ortumcore.sat_shift_add %lhs, %rhs {shift = 4 : i64} : (i32, i32) -> i32
  return %0 : i32
}

// -----

func.func @scaled_sub_shift_below_range(%lhs: i32, %rhs: i32) -> i32 {
  // expected-error @+1 {{scaled saturating add/sub shift must lie in [0, 3]}}
  %0 = ortumcore.sat_shift_sub %lhs, %rhs {shift = -1 : i64} : (i32, i32) -> i32
  return %0 : i32
}
