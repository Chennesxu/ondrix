// RUN: ondrix-opt %s -split-input-file -verify-diagnostics

func.func @rejects_wrong_ranks(
    %input: tensor<4xi16>, %coeffs: tensor<5xi16>,
    %scales: tensor<1xi16>, %state: tensor<1x2xi16>) {
  // expected-error @+1 {{requires rank-1 input/scales and rank-2 coefficients/state tensors}}
  %output, %next = ondrix.sos_filter_df2_fixed %input, %coeffs, %scales, %state {
    accumulator = !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    output_overflow = #ondsp.overflow<saturate>, output_rounding = #ondsp.rounding<nearest_even>,
    product = #ondsp.product<full>, state_overflow = #ondsp.overflow<saturate>,
    state_rounding = #ondsp.rounding<nearest_even>
  } : (tensor<4xi16>, tensor<5xi16>, tensor<1xi16>, tensor<1x2xi16>)
      -> (tensor<4xi16>, tensor<1x2xi16>)
  return
}

// -----

func.func @rejects_empty_cascade(
    %input: tensor<4xi16>, %coeffs: tensor<0x5xi16>,
    %scales: tensor<0xi16>, %state: tensor<0x2xi16>) {
  // expected-error @+1 {{requires at least one second-order section}}
  %output, %next = ondrix.sos_filter_df2_fixed %input, %coeffs, %scales, %state {
    accumulator = !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    output_overflow = #ondsp.overflow<saturate>, output_rounding = #ondsp.rounding<nearest_even>,
    product = #ondsp.product<full>, state_overflow = #ondsp.overflow<saturate>,
    state_rounding = #ondsp.rounding<nearest_even>
  } : (tensor<4xi16>, tensor<0x5xi16>, tensor<0xi16>, tensor<0x2xi16>)
      -> (tensor<4xi16>, tensor<0x2xi16>)
  return
}

// -----

func.func @rejects_section_count_mismatch(
    %input: tensor<4xi16>, %coeffs: tensor<2x5xi16>,
    %scales: tensor<1xi16>, %state: tensor<2x2xi16>) {
  // expected-error @+1 {{coefficient, scale, and state section counts must match}}
  %output, %next = ondrix.sos_filter_df2_fixed %input, %coeffs, %scales, %state {
    accumulator = !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    output_overflow = #ondsp.overflow<saturate>, output_rounding = #ondsp.rounding<nearest_even>,
    product = #ondsp.product<full>, state_overflow = #ondsp.overflow<saturate>,
    state_rounding = #ondsp.rounding<nearest_even>
  } : (tensor<4xi16>, tensor<2x5xi16>, tensor<1xi16>, tensor<2x2xi16>)
      -> (tensor<4xi16>, tensor<2x2xi16>)
  return
}

// -----

func.func @rejects_element_mismatch(
    %input: tensor<4xi16>, %coeffs: tensor<2x5xi16>,
    %scales: tensor<2xi32>, %state: tensor<2x2xi16>) {
  // expected-error @+1 {{input, coefficients, scales, state, and results must match numeric storage}}
  %output, %next = ondrix.sos_filter_df2_fixed %input, %coeffs, %scales, %state {
    accumulator = !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    output_overflow = #ondsp.overflow<saturate>, output_rounding = #ondsp.rounding<nearest_even>,
    product = #ondsp.product<full>, state_overflow = #ondsp.overflow<saturate>,
    state_rounding = #ondsp.rounding<nearest_even>
  } : (tensor<4xi16>, tensor<2x5xi16>, tensor<2xi32>, tensor<2x2xi16>)
      -> (tensor<4xi16>, tensor<2x2xi16>)
  return
}

// -----

func.func @rejects_raw_high(
    %input: tensor<4xi32>, %coeffs: tensor<2x5xi32>,
    %scales: tensor<2xi32>, %state: tensor<2x2xi32>) {
  // expected-error @+1 {{supports only exact full products}}
  %output, %next = ondrix.sos_filter_df2_fixed %input, %coeffs, %scales, %state {
    accumulator = !ondsp.acc<storage = i64, frac = 30, signed, update_overflow = saturate>,
    numeric = #ondsp.fixed<signed, storage = i32, frac = 31>,
    output_overflow = #ondsp.overflow<saturate>, output_rounding = #ondsp.rounding<nearest_even>,
    product = #ondsp.product<high_raw>, state_overflow = #ondsp.overflow<saturate>,
    state_rounding = #ondsp.rounding<nearest_even>
  } : (tensor<4xi32>, tensor<2x5xi32>, tensor<2xi32>, tensor<2x2xi32>)
      -> (tensor<4xi32>, tensor<2x2xi32>)
  return
}

// -----

func.func @rejects_wrong_q15_accumulator(
    %input: tensor<4xi16>, %coeffs: tensor<2x5xi16>,
    %scales: tensor<2xi16>, %state: tensor<2x2xi16>) {
  // expected-error @+1 {{supports only signed Q15/full with i40/frac30 accumulator or signed Q31/full with i64/frac62 accumulator}}
  %output, %next = ondrix.sos_filter_df2_fixed %input, %coeffs, %scales, %state {
    accumulator = !ondsp.acc<storage = i64, frac = 30, signed, update_overflow = saturate>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    output_overflow = #ondsp.overflow<saturate>, output_rounding = #ondsp.rounding<nearest_even>,
    product = #ondsp.product<full>, state_overflow = #ondsp.overflow<saturate>,
    state_rounding = #ondsp.rounding<nearest_even>
  } : (tensor<4xi16>, tensor<2x5xi16>, tensor<2xi16>, tensor<2x2xi16>)
      -> (tensor<4xi16>, tensor<2x2xi16>)
  return
}

// -----

func.func @rejects_unsigned_profile(
    %input: tensor<4xi16>, %coeffs: tensor<2x5xi16>,
    %scales: tensor<2xi16>, %state: tensor<2x2xi16>) {
  // expected-error @+1 {{fixed product semantics currently require a signed numeric policy}}
  %output, %next = ondrix.sos_filter_df2_fixed %input, %coeffs, %scales, %state {
    accumulator = !ondsp.acc<storage = i40, frac = 30, unsigned, update_overflow = saturate>,
    numeric = #ondsp.fixed<unsigned, storage = i16, frac = 15>,
    output_overflow = #ondsp.overflow<saturate>, output_rounding = #ondsp.rounding<nearest_even>,
    product = #ondsp.product<full>, state_overflow = #ondsp.overflow<saturate>,
    state_rounding = #ondsp.rounding<nearest_even>
  } : (tensor<4xi16>, tensor<2x5xi16>, tensor<2xi16>, tensor<2x2xi16>)
      -> (tensor<4xi16>, tensor<2x2xi16>)
  return
}

// -----

func.func @rejects_other_signed_q_format(
    %input: tensor<4xi8>, %coeffs: tensor<2x5xi8>,
    %scales: tensor<2xi8>, %state: tensor<2x2xi8>) {
  // expected-error @+1 {{supports only signed Q15/full with i40/frac30 accumulator or signed Q31/full with i64/frac62 accumulator}}
  %output, %next = ondrix.sos_filter_df2_fixed %input, %coeffs, %scales, %state {
    accumulator = !ondsp.acc<storage = i32, frac = 14, signed, update_overflow = saturate>,
    numeric = #ondsp.fixed<signed, storage = i8, frac = 7>,
    output_overflow = #ondsp.overflow<saturate>, output_rounding = #ondsp.rounding<nearest_even>,
    product = #ondsp.product<full>, state_overflow = #ondsp.overflow<saturate>,
    state_rounding = #ondsp.rounding<nearest_even>
  } : (tensor<4xi8>, tensor<2x5xi8>, tensor<2xi8>, tensor<2x2xi8>)
      -> (tensor<4xi8>, tensor<2x2xi8>)
  return
}

// -----

func.func @rejects_wrong_q31_accumulator(
    %input: tensor<4xi32>, %coeffs: tensor<2x5xi32>,
    %scales: tensor<2xi32>, %state: tensor<2x2xi32>) {
  // expected-error @+1 {{supports only signed Q15/full with i40/frac30 accumulator or signed Q31/full with i64/frac62 accumulator}}
  %output, %next = ondrix.sos_filter_df2_fixed %input, %coeffs, %scales, %state {
    accumulator = !ondsp.acc<storage = i64, frac = 61, signed, update_overflow = wrap>,
    numeric = #ondsp.fixed<signed, storage = i32, frac = 31>,
    output_overflow = #ondsp.overflow<saturate>,
    output_rounding = #ondsp.rounding<toward_negative>, product = #ondsp.product<full>,
    state_overflow = #ondsp.overflow<wrap>, state_rounding = #ondsp.rounding<toward_zero>
  } : (tensor<4xi32>, tensor<2x5xi32>, tensor<2xi32>, tensor<2x2xi32>)
      -> (tensor<4xi32>, tensor<2x2xi32>)
  return
}

// -----

func.func @rejects_tensor_encoding(
    %input: tensor<4xi16, "encoded">, %coeffs: tensor<2x5xi16>,
    %scales: tensor<2xi16>, %state: tensor<2x2xi16>) {
  // expected-error @+1 {{does not support encoded tensor types}}
  %output, %next = ondrix.sos_filter_df2_fixed %input, %coeffs, %scales, %state {
    accumulator = !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    output_overflow = #ondsp.overflow<saturate>, output_rounding = #ondsp.rounding<nearest_even>,
    product = #ondsp.product<full>, state_overflow = #ondsp.overflow<saturate>,
    state_rounding = #ondsp.rounding<nearest_even>
  } : (tensor<4xi16, "encoded">, tensor<2x5xi16>, tensor<2xi16>, tensor<2x2xi16>)
      -> (tensor<4xi16, "encoded">, tensor<2x2xi16>)
  return
}
