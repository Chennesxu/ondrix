// RUN: ondrix-opt %s -split-input-file -verify-diagnostics

// The complete refusal set for the accumulator lane parameter at the dialect
// level. A defaulted type parameter turns every existing "is this an
// accumulator?" test into an implicit acceptor, so each consumer that has no
// per-lane meaning refuses lanes > 1 explicitly and is pinned here.

// expected-error@+1 {{accumulator must declare at least one lane}}
func.func private @zero_lanes_is_not_an_accumulator() -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate, lanes = 0>

// -----

// Importing one scalar value into W independent accumulators has no definition.
func.func @acc_import_rejects_lanes(%input: i16)
    -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate, lanes = 8> {
  // expected-error@+1 {{acc_import requires a single-lane accumulator; lanes > 1 is accepted only by acc_zero, mac, and acc_export}}
  %accumulator = ondsp.acc_import %input {
    src = #ondsp.fixed<signed, storage = i16, frac = 15>
  } : (i16) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate, lanes = 8>
  return %accumulator
      : !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate, lanes = 8>
}

// -----

// `mac` is the only multi-lane accumulator update; `mac_sub` is not part of the
// batching profile even though it shares the operation shape.
func.func @mac_sub_rejects_lanes(
    %accumulator: !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate, lanes = 8>,
    %value: vector<8xi16>, %coefficient: i16)
    -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate, lanes = 8> {
  // expected-error@+1 {{mac_sub requires a single-lane accumulator; lanes > 1 is accepted only by acc_zero, mac, and acc_export}}
  %result = ondsp.mac_sub %accumulator, %value, %coefficient {
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    product = #ondsp.product<full>
  } : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate, lanes = 8>,
       vector<8xi16>, i16)
      -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate, lanes = 8>
  return %result
      : !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate, lanes = 8>
}

// -----

// A pre-computed term is the normalization point for horizontal partial sums,
// whose lane meaning is the reduction axis, not independent outputs.
func.func @acc_add_term_rejects_lanes(
    %accumulator: !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate, lanes = 8>,
    %term: i40)
    -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate, lanes = 8> {
  // expected-error@+1 {{acc_add_term requires a single-lane accumulator; lanes > 1 is accepted only by acc_zero, mac, and acc_export}}
  %result = ondsp.acc_add_term %accumulator, %term {
    term_numeric = #ondsp.fixed<signed, storage = i40, frac = 30>
  } : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate, lanes = 8>, i40)
      -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate, lanes = 8>
  return %result
      : !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate, lanes = 8>
}

// -----

// A reduction spends its lanes on the reduction axis. Accepting a multi-lane
// accumulator here would mean two incompatible readings of the same lanes, so
// the ordered reduction refuses them and the horizontal vectorization passes
// consequently never see one.
func.func @reduce_mac_rejects_lanes(%lhs: memref<8xi16>, %rhs: memref<8xi16>)
    -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate, lanes = 8> {
  %zero = ondsp.acc_zero
      : !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate, lanes = 8>
  // expected-error@+1 {{reduce_mac requires a single-lane accumulator; lanes > 1 is accepted only by acc_zero, mac, and acc_export}}
  %result = ondsp.reduce_mac %zero, %lhs, %rhs {
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    product = #ondsp.product<full>
  } : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate, lanes = 8>,
       memref<8xi16>, memref<8xi16>)
      -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate, lanes = 8>
  return %result
      : !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate, lanes = 8>
}

// -----

// The coefficient is scalar in every lane profile: a vector coefficient would
// be a different operation, so the signature refuses it outright.
func.func @mac_rejects_vector_coefficient(
    %accumulator: !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate, lanes = 8>,
    %value: vector<8xi16>)
    -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate, lanes = 8> {
  // expected-error@+1 {{operand #2 must be integer, but got 'vector<8xi16>'}}
  %result = ondsp.mac %accumulator, %value, %value {
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    product = #ondsp.product<full>
  } : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate, lanes = 8>,
       vector<8xi16>, vector<8xi16>)
      -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate, lanes = 8>
  return %result
      : !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate, lanes = 8>
}

// -----

func.func @mac_rejects_lane_count_mismatch(
    %accumulator: !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate, lanes = 8>,
    %value: vector<4xi16>, %coefficient: i16)
    -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate, lanes = 8> {
  // expected-error@+1 {{value lane count 4 does not match accumulator lanes 8}}
  %result = ondsp.mac %accumulator, %value, %coefficient {
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    product = #ondsp.product<full>
  } : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate, lanes = 8>,
       vector<4xi16>, i16)
      -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate, lanes = 8>
  return %result
      : !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate, lanes = 8>
}

// -----

// A single-lane accumulator must not silently accept a vector value either:
// that would be a batched update wearing an unbatched type.
func.func @single_lane_mac_rejects_vector_value(
    %accumulator: !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>,
    %value: vector<8xi16>, %coefficient: i16)
    -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate> {
  // expected-error@+1 {{value must be a scalar value for a single-lane accumulator}}
  %result = ondsp.mac %accumulator, %value, %coefficient {
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    product = #ondsp.product<full>
  } : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>,
       vector<8xi16>, i16)
      -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  return %result : !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
}

// -----

func.func @mac_rejects_scalable_value(
    %accumulator: !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate, lanes = 8>,
    %value: vector<[8]xi16>, %coefficient: i16)
    -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate, lanes = 8> {
  // expected-error@+1 {{value must be a fixed-length rank-1 vector value for a multi-lane accumulator}}
  %result = ondsp.mac %accumulator, %value, %coefficient {
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    product = #ondsp.product<full>
  } : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate, lanes = 8>,
       vector<[8]xi16>, i16)
      -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate, lanes = 8>
  return %result
      : !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate, lanes = 8>
}

// -----

// Exporting W lanes into one destination element would be a horizontal
// reduction the operation does not declare.
func.func @acc_export_rejects_scalar_result(
    %accumulator: !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate, lanes = 8>)
    -> i16 {
  // expected-error@+1 {{result must be a fixed-length rank-1 vector value for a multi-lane accumulator}}
  %result = ondsp.acc_export %accumulator {
    dst = #ondsp.fixed<signed, storage = i16, frac = 15>,
    rounding = #ondsp.rounding<nearest_even>,
    overflow = #ondsp.overflow<saturate>
  } : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate, lanes = 8>) -> i16
  return %result : i16
}

// -----

func.func @acc_export_rejects_lane_count_mismatch(
    %accumulator: !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate, lanes = 8>)
    -> vector<4xi16> {
  // expected-error@+1 {{result lane count 4 does not match accumulator lanes 8}}
  %result = ondsp.acc_export %accumulator {
    dst = #ondsp.fixed<signed, storage = i16, frac = 15>,
    rounding = #ondsp.rounding<nearest_even>,
    overflow = #ondsp.overflow<saturate>
  } : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate, lanes = 8>)
      -> vector<4xi16>
  return %result : vector<4xi16>
}

// -----

// A one-element vector denotes the same values as a scalar but is a shape no
// existing lowering handles, so the single-lane form stays exactly scalar.
func.func @single_lane_mac_rejects_one_element_vector(
    %accumulator: !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>,
    %value: vector<1xi16>, %coefficient: i16)
    -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate> {
  // expected-error@+1 {{value must be a scalar value for a single-lane accumulator}}
  %result = ondsp.mac %accumulator, %value, %coefficient {
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    product = #ondsp.product<full>
  } : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>,
       vector<1xi16>, i16)
      -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  return %result : !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
}

// -----

func.func @single_lane_export_rejects_one_element_vector(
    %accumulator: !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>)
    -> vector<1xi16> {
  // expected-error@+1 {{result must be a scalar value for a single-lane accumulator}}
  %result = ondsp.acc_export %accumulator {
    dst = #ondsp.fixed<signed, storage = i16, frac = 15>,
    rounding = #ondsp.rounding<nearest_even>,
    overflow = #ondsp.overflow<saturate>
  } : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>) -> vector<1xi16>
  return %result : vector<1xi16>
}
