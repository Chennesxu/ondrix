// RUN: ondrix-opt %s -split-input-file -verify-diagnostics

func.func @requires_factor_two(
    %input: tensor<4xi16>, %coeffs: tensor<3xi16>, %init: tensor<12xi16>) {
  // expected-error @+1 {{first executable profile requires factor 2}}
  %0 = ondrix.fir_interpolate %input, %coeffs, %init {
    accumulator = !ondsp.acc<storage = i40, frac = 30, signed,
                              update_overflow = saturate>,
    dst = #ondsp.fixed<signed, storage = i16, frac = 15>,
    factor = 3,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    overflow = #ondsp.overflow<saturate>,
    product = #ondsp.product<full>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<4xi16>, tensor<3xi16>, tensor<12xi16>) -> tensor<12xi16>
  return
}

// -----

func.func @requires_nonempty_input(
    %input: tensor<0xi16>, %coeffs: tensor<3xi16>, %init: tensor<1xi16>) {
  // expected-error @+1 {{requires at least one input sample}}
  %0 = ondrix.fir_interpolate %input, %coeffs, %init {
    accumulator = !ondsp.acc<storage = i40, frac = 30, signed,
                              update_overflow = saturate>,
    dst = #ondsp.fixed<signed, storage = i16, frac = 15>,
    factor = 2,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    overflow = #ondsp.overflow<saturate>,
    product = #ondsp.product<full>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<0xi16>, tensor<3xi16>, tensor<1xi16>) -> tensor<1xi16>
  return
}

// -----

func.func @requires_nonempty_coefficients(
    %input: tensor<4xi16>, %coeffs: tensor<0xi16>, %init: tensor<7xi16>) {
  // expected-error @+1 {{requires at least one coefficient}}
  %0 = ondrix.fir_interpolate %input, %coeffs, %init {
    accumulator = !ondsp.acc<storage = i40, frac = 30, signed,
                              update_overflow = saturate>,
    dst = #ondsp.fixed<signed, storage = i16, frac = 15>,
    factor = 2,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    overflow = #ondsp.overflow<saturate>,
    product = #ondsp.product<full>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<4xi16>, tensor<0xi16>, tensor<7xi16>) -> tensor<7xi16>
  return
}

// -----

func.func @rejects_static_extent_overflow(
    %input: tensor<9223372036854775807xi16>, %coeffs: tensor<2xi16>,
    %init: tensor<1xi16>) {
  // expected-error @+1 {{result length exceeds the indexable extent range}}
  %0 = ondrix.fir_interpolate %input, %coeffs, %init {
    accumulator = !ondsp.acc<storage = i40, frac = 30, signed,
                              update_overflow = saturate>,
    dst = #ondsp.fixed<signed, storage = i16, frac = 15>,
    factor = 2,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    overflow = #ondsp.overflow<saturate>,
    product = #ondsp.product<full>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<9223372036854775807xi16>, tensor<2xi16>, tensor<1xi16>)
      -> tensor<1xi16>
  return
}

// -----

func.func @requires_exact_result_length(
    %input: tensor<4xi16>, %coeffs: tensor<3xi16>, %init: tensor<8xi16>) {
  // expected-error @+1 {{result length must be 9}}
  %0 = ondrix.fir_interpolate %input, %coeffs, %init {
    accumulator = !ondsp.acc<storage = i40, frac = 30, signed,
                              update_overflow = saturate>,
    dst = #ondsp.fixed<signed, storage = i16, frac = 15>,
    factor = 2,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    overflow = #ondsp.overflow<saturate>,
    product = #ondsp.product<full>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<4xi16>, tensor<3xi16>, tensor<8xi16>) -> tensor<8xi16>
  return
}

// -----

func.func @rejects_encoded_result(
    %input: tensor<4xi16>, %coeffs: tensor<3xi16>,
    %init: tensor<9xi16, "encoded">) {
  // expected-error @+1 {{does not support encoded tensor types}}
  %0 = ondrix.fir_interpolate %input, %coeffs, %init {
    accumulator = !ondsp.acc<storage = i40, frac = 30, signed,
                              update_overflow = saturate>,
    dst = #ondsp.fixed<signed, storage = i16, frac = 15>,
    factor = 2,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    overflow = #ondsp.overflow<saturate>,
    product = #ondsp.product<full>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<4xi16>, tensor<3xi16>, tensor<9xi16, "encoded">)
      -> tensor<9xi16, "encoded">
  return
}

// -----

func.func @rejects_q31_profile(
    %input: tensor<4xi32>, %coeffs: tensor<3xi32>, %init: tensor<9xi32>) {
  // expected-error @+1 {{supports only signed Q15/full with a signed frac30 accumulator of at least 32 bits}}
  %0 = ondrix.fir_interpolate %input, %coeffs, %init {
    accumulator = !ondsp.acc<storage = i64, frac = 62, signed,
                              update_overflow = saturate>,
    dst = #ondsp.fixed<signed, storage = i32, frac = 31>,
    factor = 2,
    numeric = #ondsp.fixed<signed, storage = i32, frac = 31>,
    overflow = #ondsp.overflow<saturate>,
    product = #ondsp.product<full>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<4xi32>, tensor<3xi32>, tensor<9xi32>) -> tensor<9xi32>
  return
}

// -----

// A newly declared dialect rounding mode does not silently widen this
// contract: the admissible tie rules are per operation, and nearest_ties_
// positive has no lowering or differential evidence here.

func.func @rejects_unestablished_rounding(
    %input: tensor<4xi16>, %coeffs: tensor<3xi16>, %init: tensor<9xi16>) {
  // expected-error @+1 {{Q15 resampling supports toward_negative, toward_zero, or nearest_even rounding}}
  %0 = ondrix.fir_interpolate %input, %coeffs, %init {
    accumulator = !ondsp.acc<storage = i40, frac = 30, signed,
                              update_overflow = saturate>,
    dst = #ondsp.fixed<signed, storage = i16, frac = 15>,
    factor = 2,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    overflow = #ondsp.overflow<saturate>,
    product = #ondsp.product<full>,
    rounding = #ondsp.rounding<nearest_ties_positive>
  } : (tensor<4xi16>, tensor<3xi16>, tensor<9xi16>) -> tensor<9xi16>
  return
}
