// RUN: ondrix-opt %s --split-input-file --verify-diagnostics

func.func @zero_divisor(%x: i32) -> i16 {
  // expected-error@+1 {{round_div requires a positive static divisor}}
  %y = ondsp.round_div %x {
    divisor = 0 : i64, pre_shift_left = 0 : i64,
    rounding = #ondsp.rounding<nearest_even>,
    overflow = #ondsp.overflow<saturate>
  } : (i32) -> i16
  return %y : i16
}

// -----

func.func @negative_divisor(%x: i32) -> i16 {
  // expected-error@+1 {{round_div requires a positive static divisor}}
  %y = ondsp.round_div %x {
    divisor = -3 : i64, pre_shift_left = 0 : i64,
    rounding = #ondsp.rounding<nearest_even>,
    overflow = #ondsp.overflow<saturate>
  } : (i32) -> i16
  return %y : i16
}

// -----

func.func @pre_shift_out_of_range(%x: i32) -> i16 {
  // expected-error@+1 {{pre_shift_left must lie in [0, 63]}}
  %y = ondsp.round_div %x {
    divisor = 3 : i64, pre_shift_left = 64 : i64,
    rounding = #ondsp.rounding<nearest_even>,
    overflow = #ondsp.overflow<saturate>
  } : (i32) -> i16
  return %y : i16
}

// -----

func.func @carrier_overflow(%x: i128) -> i16 {
  // expected-error@+1 {{the exact scaled carrier (input width + pre_shift_left) must not exceed 128 bits}}
  %y = ondsp.round_div %x {
    divisor = 3 : i64, pre_shift_left = 1 : i64,
    rounding = #ondsp.rounding<nearest_even>,
    overflow = #ondsp.overflow<saturate>
  } : (i128) -> i16
  return %y : i16
}

// -----

func.func @unrepresentable_divisor(%x: i16) -> i16 {
  // expected-error@+1 {{the divisor must be representable in the exact scaled carrier}}
  %y = ondsp.round_div %x {
    divisor = 40000 : i64, pre_shift_left = 0 : i64,
    rounding = #ondsp.rounding<nearest_even>,
    overflow = #ondsp.overflow<saturate>
  } : (i16) -> i16
  return %y : i16
}

// -----

func.func @widening_result(%x: i16) -> i32 {
  // expected-error@+1 {{round_div does not widen: the result storage must not exceed the exact scaled carrier}}
  %y = ondsp.round_div %x {
    divisor = 3 : i64, pre_shift_left = 0 : i64,
    rounding = #ondsp.rounding<nearest_even>,
    overflow = #ondsp.overflow<saturate>
  } : (i16) -> i32
  return %y : i32
}

// -----

func.func @shape_mismatch(%x: vector<4xi32>) -> vector<8xi16> {
  // expected-error@+1 {{operands and results must use the same scalar or static shaped domain}}
  %y = ondsp.round_div %x {
    divisor = 3 : i64, pre_shift_left = 0 : i64,
    rounding = #ondsp.rounding<nearest_even>,
    overflow = #ondsp.overflow<saturate>
  } : (vector<4xi32>) -> vector<8xi16>
  return %y : vector<8xi16>
}
