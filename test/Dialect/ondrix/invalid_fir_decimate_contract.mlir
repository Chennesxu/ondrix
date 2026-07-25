// RUN: ondrix-opt %s -split-input-file -verify-diagnostics

func.func @requires_decimation(
    %input: tensor<12xi16>, %coeffs: tensor<5xi16>, %init: tensor<8xi16>) {
  // expected-error @+1 {{requires factor at least 2}}
  %0 = ondrix.fir_decimate %input, %coeffs, %init {
    accumulator = !ondsp.acc<storage = i40, frac = 30, signed,
                              update_overflow = saturate>,
    dst = #ondsp.fixed<signed, storage = i16, frac = 15>,
    factor = 1,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    overflow = #ondsp.overflow<saturate>,
    product = #ondsp.product<full>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<12xi16>, tensor<5xi16>, tensor<8xi16>) -> tensor<8xi16>
  return
}

// -----

func.func @rejects_negative_factor(
    %input: tensor<12xi16>, %coeffs: tensor<5xi16>, %init: tensor<1xi16>) {
  // expected-error @+1 {{requires factor at least 2}}
  %0 = ondrix.fir_decimate %input, %coeffs, %init {
    accumulator = !ondsp.acc<storage = i40, frac = 30, signed,
                              update_overflow = saturate>,
    dst = #ondsp.fixed<signed, storage = i16, frac = 15>,
    factor = -1,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    overflow = #ondsp.overflow<saturate>,
    product = #ondsp.product<full>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<12xi16>, tensor<5xi16>, tensor<1xi16>) -> tensor<1xi16>
  return
}

// -----

func.func @requires_exact_result_length(
    %input: tensor<12xi16>, %coeffs: tensor<5xi16>, %init: tensor<5xi16>) {
  // expected-error @+1 {{result length must be 4}}
  %0 = ondrix.fir_decimate %input, %coeffs, %init {
    accumulator = !ondsp.acc<storage = i40, frac = 30, signed,
                              update_overflow = saturate>,
    dst = #ondsp.fixed<signed, storage = i16, frac = 15>,
    factor = 2,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    overflow = #ondsp.overflow<saturate>,
    product = #ondsp.product<full>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<12xi16>, tensor<5xi16>, tensor<5xi16>) -> tensor<5xi16>
  return
}

// -----

func.func @rejects_encoded_input(
    %input: tensor<12xi16, "encoded">, %coeffs: tensor<5xi16>,
    %init: tensor<4xi16>) {
  // expected-error @+1 {{does not support encoded tensor types}}
  %0 = ondrix.fir_decimate %input, %coeffs, %init {
    accumulator = !ondsp.acc<storage = i40, frac = 30, signed,
                              update_overflow = saturate>,
    dst = #ondsp.fixed<signed, storage = i16, frac = 15>,
    factor = 2,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    overflow = #ondsp.overflow<saturate>,
    product = #ondsp.product<full>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<12xi16, "encoded">, tensor<5xi16>, tensor<4xi16>)
      -> tensor<4xi16>
  return
}

// -----

func.func @rejects_high_raw_product(
    %input: tensor<12xi16>, %coeffs: tensor<5xi16>, %init: tensor<4xi16>) {
  // expected-error @+1 {{supports only signed Q15/full with a signed frac30 accumulator of at least 32 bits}}
  %0 = ondrix.fir_decimate %input, %coeffs, %init {
    accumulator = !ondsp.acc<storage = i40, frac = 30, signed,
                              update_overflow = saturate>,
    dst = #ondsp.fixed<signed, storage = i16, frac = 15>,
    factor = 2,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    overflow = #ondsp.overflow<saturate>,
    product = #ondsp.product<high_raw>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<12xi16>, tensor<5xi16>, tensor<4xi16>) -> tensor<4xi16>
  return
}

// -----

func.func @rejects_q31_profile(
    %input: tensor<12xi32>, %coeffs: tensor<5xi32>, %init: tensor<4xi32>) {
  // expected-error @+1 {{supports only signed Q15/full with a signed frac30 accumulator of at least 32 bits}}
  %0 = ondrix.fir_decimate %input, %coeffs, %init {
    accumulator = !ondsp.acc<storage = i64, frac = 62, signed,
                              update_overflow = saturate>,
    dst = #ondsp.fixed<signed, storage = i32, frac = 31>,
    factor = 2,
    numeric = #ondsp.fixed<signed, storage = i32, frac = 31>,
    overflow = #ondsp.overflow<saturate>,
    product = #ondsp.product<full>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<12xi32>, tensor<5xi32>, tensor<4xi32>) -> tensor<4xi32>
  return
}
