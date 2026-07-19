// RUN: ondrix-opt %s -split-input-file -verify-diagnostics

func.func @requires_rank_one(
    %input: tensor<2x4xi16>, %coeffs: tensor<3xi16>, %init: tensor<6xi16>) {
  // expected-error @+1 {{requires rank-1 input, coefficient, and init tensors}}
  %0 = ondrix.fir_filter %input, %coeffs, %init {
    accumulator = !ondsp.acc<storage = i40, frac = 30, signed,
                              update_overflow = saturate>,
    boundary = #ondrix.fir_boundary<valid>,
    dst = #ondsp.fixed<signed, storage = i16, frac = 15>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    overflow = #ondsp.overflow<saturate>,
    product = #ondsp.product<full>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<2x4xi16>, tensor<3xi16>, tensor<6xi16>) -> tensor<6xi16>
  return
}

// -----

func.func @requires_nonempty_coefficients(
    %input: tensor<8xi16>, %coeffs: tensor<0xi16>, %init: tensor<9xi16>) {
  // expected-error @+1 {{valid FIR requires at least one coefficient}}
  %0 = ondrix.fir_filter %input, %coeffs, %init {
    accumulator = !ondsp.acc<storage = i40, frac = 30, signed,
                              update_overflow = saturate>,
    boundary = #ondrix.fir_boundary<valid>,
    dst = #ondsp.fixed<signed, storage = i16, frac = 15>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    overflow = #ondsp.overflow<saturate>,
    product = #ondsp.product<full>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<8xi16>, tensor<0xi16>, tensor<9xi16>) -> tensor<9xi16>
  return
}

// -----

func.func @requires_covered_window(
    %input: tensor<2xi16>, %coeffs: tensor<3xi16>, %init: tensor<1xi16>) {
  // expected-error @+1 {{valid FIR input must cover one coefficient window}}
  %0 = ondrix.fir_filter %input, %coeffs, %init {
    accumulator = !ondsp.acc<storage = i40, frac = 30, signed,
                              update_overflow = saturate>,
    boundary = #ondrix.fir_boundary<valid>,
    dst = #ondsp.fixed<signed, storage = i16, frac = 15>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    overflow = #ondsp.overflow<saturate>,
    product = #ondsp.product<full>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<2xi16>, tensor<3xi16>, tensor<1xi16>) -> tensor<1xi16>
  return
}

// -----

func.func @requires_exact_output_length(
    %input: tensor<8xi16>, %coeffs: tensor<3xi16>, %init: tensor<5xi16>) {
  // expected-error @+1 {{valid FIR output length must be 6}}
  %0 = ondrix.fir_filter %input, %coeffs, %init {
    accumulator = !ondsp.acc<storage = i40, frac = 30, signed,
                              update_overflow = saturate>,
    boundary = #ondrix.fir_boundary<valid>,
    dst = #ondsp.fixed<signed, storage = i16, frac = 15>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    overflow = #ondsp.overflow<saturate>,
    product = #ondsp.product<full>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<8xi16>, tensor<3xi16>, tensor<5xi16>) -> tensor<5xi16>
  return
}

// -----

func.func @requires_explicit_fixed_lifecycle(
    %input: tensor<8xi16>, %coeffs: tensor<3xi16>, %init: tensor<6xi16>) {
  // expected-error @+1 {{fixed FIR filter requires accumulator, dst, rounding, and overflow attributes}}
  %0 = ondrix.fir_filter %input, %coeffs, %init {
    boundary = #ondrix.fir_boundary<valid>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    product = #ondsp.product<full>
  } : (tensor<8xi16>, tensor<3xi16>, tensor<6xi16>) -> tensor<6xi16>
  return
}

// -----

func.func @requires_matching_accumulator_frac(
    %input: tensor<8xi16>, %coeffs: tensor<3xi16>, %init: tensor<6xi16>) {
  // expected-error @+1 {{accumulator frac 29 does not match product frac 30}}
  %0 = ondrix.fir_filter %input, %coeffs, %init {
    accumulator = !ondsp.acc<storage = i40, frac = 29, signed,
                              update_overflow = saturate>,
    boundary = #ondrix.fir_boundary<valid>,
    dst = #ondsp.fixed<signed, storage = i16, frac = 15>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    overflow = #ondsp.overflow<saturate>,
    product = #ondsp.product<full>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<8xi16>, tensor<3xi16>, tensor<6xi16>) -> tensor<6xi16>
  return
}

// -----

func.func @requires_matching_destination_storage(
    %input: tensor<8xi16>, %coeffs: tensor<3xi16>, %init: tensor<6xi32>) {
  // expected-error @+1 {{init and result element type must match destination storage}}
  %0 = ondrix.fir_filter %input, %coeffs, %init {
    accumulator = !ondsp.acc<storage = i40, frac = 30, signed,
                              update_overflow = saturate>,
    boundary = #ondrix.fir_boundary<valid>,
    dst = #ondsp.fixed<signed, storage = i16, frac = 15>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    overflow = #ondsp.overflow<saturate>,
    product = #ondsp.product<full>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<8xi16>, tensor<3xi16>, tensor<6xi32>) -> tensor<6xi32>
  return
}

// -----

func.func @fp_rejects_fixed_lifecycle(
    %input: tensor<8xf32>, %coeffs: tensor<3xf32>, %init: tensor<6xf32>) {
  // expected-error @+1 {{floating-point FIR filter must not specify fixed-point accumulator or export policy}}
  %0 = ondrix.fir_filter %input, %coeffs, %init {
    accumulator = !ondsp.acc<storage = i40, frac = 30, signed,
                              update_overflow = saturate>,
    boundary = #ondrix.fir_boundary<valid>,
    numeric = #ondsp.fp<format = f32, contract = fma>
  } : (tensor<8xf32>, tensor<3xf32>, tensor<6xf32>) -> tensor<6xf32>
  return
}

// -----

func.func @fp_requires_matching_format(
    %input: tensor<8xf64>, %coeffs: tensor<3xf64>, %init: tensor<6xf32>) {
  // expected-error @+1 {{floating-point input, coefficients, init, and result must match format}}
  %0 = ondrix.fir_filter %input, %coeffs, %init {
    boundary = #ondrix.fir_boundary<valid>,
    numeric = #ondsp.fp<format = f32, contract = fma>
  } : (tensor<8xf64>, tensor<3xf64>, tensor<6xf32>) -> tensor<6xf32>
  return
}
