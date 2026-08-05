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

// -----

// The export policy exists to place a fixed-point requantization boundary;
// an f32 reduction has none, so carrying one is a declaration error rather
// than an ignorable attribute.

func.func @rejects_fp_export_policy(
    %input: tensor<12xf32>, %coeffs: tensor<5xf32>, %init: tensor<4xf32>) {
  // expected-error @+1 {{floating-point resampling must not specify a fixed-point export policy}}
  %0 = ondrix.fir_decimate %input, %coeffs, %init {
    dst = #ondsp.fixed<signed, storage = i16, frac = 15>,
    factor = 2,
    numeric = #ondsp.fp<format = f32, contract = off>
  } : (tensor<12xf32>, tensor<5xf32>, tensor<4xf32>) -> tensor<4xf32>
  return
}

// -----

func.func @rejects_missing_fixed_export_policy(
    %input: tensor<12xi16>, %coeffs: tensor<5xi16>, %init: tensor<4xi16>) {
  // expected-error @+1 {{fixed resampling requires accumulator, dst, rounding, and overflow attributes}}
  %0 = ondrix.fir_decimate %input, %coeffs, %init {
    factor = 2,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    product = #ondsp.product<full>
  } : (tensor<12xi16>, tensor<5xi16>, tensor<4xi16>) -> tensor<4xi16>
  return
}

// -----

func.func @rejects_f64_resampling(
    %input: tensor<12xf64>, %coeffs: tensor<5xf64>, %init: tensor<4xf64>) {
  // expected-error @+1 {{executable resampling supports the f32 floating-point format}}
  %0 = ondrix.fir_decimate %input, %coeffs, %init {
    factor = 2,
    numeric = #ondsp.fp<format = f64, contract = off>
  } : (tensor<12xf64>, tensor<5xf64>, tensor<4xf64>) -> tensor<4xf64>
  return
}
