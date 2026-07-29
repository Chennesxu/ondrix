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

// -----

func.func @full_requires_nonempty_input(
    %input: tensor<0xi16>, %coeffs: tensor<3xi16>, %init: tensor<2xi16>) {
  // expected-error @+1 {{full FIR requires at least one input sample}}
  %0 = ondrix.fir_filter %input, %coeffs, %init {
    accumulator = !ondsp.acc<storage = i40, frac = 30, signed,
                              update_overflow = saturate>,
    boundary = #ondrix.fir_boundary<full>,
    dst = #ondsp.fixed<signed, storage = i16, frac = 15>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    overflow = #ondsp.overflow<saturate>,
    product = #ondsp.product<full>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<0xi16>, tensor<3xi16>, tensor<2xi16>) -> tensor<2xi16>
  return
}

// -----

func.func @full_requires_nonempty_coefficients(
    %input: tensor<4xi16>, %coeffs: tensor<0xi16>, %init: tensor<3xi16>) {
  // expected-error @+1 {{full FIR requires at least one coefficient}}
  %0 = ondrix.fir_filter %input, %coeffs, %init {
    accumulator = !ondsp.acc<storage = i40, frac = 30, signed,
                              update_overflow = saturate>,
    boundary = #ondrix.fir_boundary<full>,
    dst = #ondsp.fixed<signed, storage = i16, frac = 15>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    overflow = #ondsp.overflow<saturate>,
    product = #ondsp.product<full>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<4xi16>, tensor<0xi16>, tensor<3xi16>) -> tensor<3xi16>
  return
}

// -----

func.func @full_requires_exact_output_length(
    %input: tensor<4xi16>, %coeffs: tensor<3xi16>, %init: tensor<5xi16>) {
  // expected-error @+1 {{full FIR output length must be 6}}
  %0 = ondrix.fir_filter %input, %coeffs, %init {
    accumulator = !ondsp.acc<storage = i40, frac = 30, signed,
                              update_overflow = saturate>,
    boundary = #ondrix.fir_boundary<full>,
    dst = #ondsp.fixed<signed, storage = i16, frac = 15>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    overflow = #ondsp.overflow<saturate>,
    product = #ondsp.product<full>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<4xi16>, tensor<3xi16>, tensor<5xi16>) -> tensor<5xi16>
  return
}

// -----

func.func @valid_rejects_output_origin(
    %input: tensor<8xi16>, %coeffs: tensor<3xi16>, %init: tensor<4xi16>) {
  %origin = arith.constant 1 : index
  // expected-error @+1 {{output_origin is supported only for full FIR boundaries}}
  %0 = ondrix.fir_filter %input, %coeffs, %init, %origin {
    accumulator = !ondsp.acc<storage = i40, frac = 30, signed,
                              update_overflow = saturate>,
    boundary = #ondrix.fir_boundary<valid>,
    dst = #ondsp.fixed<signed, storage = i16, frac = 15>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    overflow = #ondsp.overflow<saturate>,
    product = #ondsp.product<full>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<8xi16>, tensor<3xi16>, tensor<4xi16>, index) -> tensor<4xi16>
  return
}

// -----

func.func @full_rejects_out_of_range_output_tile(
    %input: tensor<4xi16>, %coeffs: tensor<3xi16>, %init: tensor<3xi16>) {
  %origin = arith.constant 4 : index
  // expected-error @+1 {{full FIR output tile exceeds the complete output range}}
  %0 = ondrix.fir_filter %input, %coeffs, %init, %origin {
    accumulator = !ondsp.acc<storage = i40, frac = 30, signed,
                              update_overflow = saturate>,
    boundary = #ondrix.fir_boundary<full>,
    dst = #ondsp.fixed<signed, storage = i16, frac = 15>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    overflow = #ondsp.overflow<saturate>,
    product = #ondsp.product<full>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<4xi16>, tensor<3xi16>, tensor<3xi16>, index) -> tensor<3xi16>
  return
}

// -----

func.func @full_rejects_negative_output_origin(
    %input: tensor<4xi16>, %coeffs: tensor<3xi16>, %init: tensor<1xi16>) {
  %origin = arith.constant -1 : index
  // expected-error @+1 {{full FIR output tile exceeds the complete output range}}
  %0 = ondrix.fir_filter %input, %coeffs, %init, %origin {
    accumulator = !ondsp.acc<storage = i40, frac = 30, signed,
                              update_overflow = saturate>,
    boundary = #ondrix.fir_boundary<full>,
    dst = #ondsp.fixed<signed, storage = i16, frac = 15>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    overflow = #ondsp.overflow<saturate>,
    product = #ondsp.product<full>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<4xi16>, tensor<3xi16>, tensor<1xi16>, index) -> tensor<1xi16>
  return
}

// -----

func.func @full_rejects_static_origin_beyond_dynamic_output(
    %input: tensor<4xf32>, %coeffs: tensor<3xf32>, %init: tensor<?xf32>) {
  %origin = arith.constant 7 : index
  // expected-error @+1 {{full FIR output tile exceeds the complete output range}}
  %0 = ondrix.fir_filter %input, %coeffs, %init, %origin {
    boundary = #ondrix.fir_boundary<full>,
    numeric = #ondsp.fp<format = f32, contract = off>
  } : (tensor<4xf32>, tensor<3xf32>, tensor<?xf32>, index) -> tensor<?xf32>
  return
}

// -----

// A newly declared dialect rounding mode does not silently widen this
// contract: the admissible tie rules are per operation, and nearest_ties_
// positive has no lowering or differential evidence here.

func.func @rejects_unestablished_rounding(
    %input: tensor<8xi16>, %coeffs: tensor<3xi16>, %init: tensor<6xi16>) {
  // expected-error @+1 {{fixed FIR filter supports toward_negative, toward_zero, or nearest_even rounding}}
  %0 = ondrix.fir_filter %input, %coeffs, %init {
    accumulator = !ondsp.acc<storage = i40, frac = 30, signed,
                              update_overflow = saturate>,
    boundary = #ondrix.fir_boundary<valid>,
    dst = #ondsp.fixed<signed, storage = i16, frac = 15>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    overflow = #ondsp.overflow<saturate>,
    product = #ondsp.product<full>,
    rounding = #ondsp.rounding<nearest_ties_positive>
  } : (tensor<8xi16>, tensor<3xi16>, tensor<6xi16>) -> tensor<6xi16>
  return
}
