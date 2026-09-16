// RUN: ondrix-opt %s --split-input-file --verify-diagnostics

func.func @policy_is_required(%x: i16, %y: i16) -> i16 {
  // expected-error @below {{requires attribute 'nonpositive'}}
  %0 = ondsp.round_quotient %x, %y {
    pre_shift_left = 15 : i64,
    rounding = #ondsp.rounding<nearest_even>,
    overflow = #ondsp.overflow<saturate>
  } : (i16, i16) -> i16
  return %0 : i16
}

// -----

func.func @scalars_only(%x: vector<4xi16>, %y: vector<4xi16>) -> vector<4xi16> {
  // expected-error @below {{round_quotient takes a scalar signless integer input, divisor and result}}
  %0 = ondsp.round_quotient %x, %y {
    pre_shift_left = 15 : i64,
    rounding = #ondsp.rounding<nearest_even>,
    overflow = #ondsp.overflow<saturate>,
    nonpositive = #ondsp.nonpositive_divisor<trap>
  } : (vector<4xi16>, vector<4xi16>) -> vector<4xi16>
  return %0 : vector<4xi16>
}

// -----

func.func @divisor_wider_than_carrier(%x: i16, %y: i64) -> i16 {
  // expected-error @below {{the divisor storage must not exceed the exact scaled carrier}}
  %0 = ondsp.round_quotient %x, %y {
    pre_shift_left = 0 : i64,
    rounding = #ondsp.rounding<nearest_even>,
    overflow = #ondsp.overflow<saturate>,
    nonpositive = #ondsp.nonpositive_divisor<trap>
  } : (i16, i64) -> i16
  return %0 : i16
}

// -----

func.func @result_wider_than_carrier(%x: i16, %y: i16) -> i64 {
  // expected-error @below {{round_quotient does not widen}}
  %0 = ondsp.round_quotient %x, %y {
    pre_shift_left = 15 : i64,
    rounding = #ondsp.rounding<nearest_even>,
    overflow = #ondsp.overflow<saturate>,
    nonpositive = #ondsp.nonpositive_divisor<saturate>
  } : (i16, i16) -> i64
  return %0 : i64
}

// -----

func.func @pre_shift_range(%x: i64, %y: i64) -> i64 {
  // expected-error @below {{pre_shift_left must lie in [0, 63]}}
  %0 = ondsp.round_quotient %x, %y {
    pre_shift_left = 64 : i64,
    rounding = #ondsp.rounding<nearest_even>,
    overflow = #ondsp.overflow<saturate>,
    nonpositive = #ondsp.nonpositive_divisor<saturate>
  } : (i64, i64) -> i64
  return %0 : i64
}
