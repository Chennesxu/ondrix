// RUN: ondrix-opt %s -split-input-file -verify-diagnostics

func.func @rejects_wrong_ranks(
    %input: tensor<4xf32>, %coeffs: tensor<5xf32>,
    %scales: tensor<1xf32>, %state: tensor<1x2xf32>) {
  // expected-error @+1 {{requires rank-1 input/scales and rank-2 coefficients/state tensors}}
  %output, %next = ondrix.sos_filter_tdf2 %input, %coeffs, %scales, %state {
    numeric = #ondsp.fp<format = f32, contract = fma>
  } : (tensor<4xf32>, tensor<5xf32>, tensor<1xf32>, tensor<1x2xf32>)
      -> (tensor<4xf32>, tensor<1x2xf32>)
  return
}

// -----

func.func @rejects_coefficient_layout(
    %input: tensor<4xf32>, %coeffs: tensor<2x6xf32>,
    %scales: tensor<2xf32>, %state: tensor<2x2xf32>) {
  // expected-error @+1 {{coefficient trailing dimension must be statically 5}}
  %output, %next = ondrix.sos_filter_tdf2 %input, %coeffs, %scales, %state {
    numeric = #ondsp.fp<format = f32, contract = fma>
  } : (tensor<4xf32>, tensor<2x6xf32>, tensor<2xf32>, tensor<2x2xf32>)
      -> (tensor<4xf32>, tensor<2x2xf32>)
  return
}

// -----

func.func @rejects_dynamic_coefficient_layout(
    %input: tensor<4xf32>, %coeffs: tensor<2x?xf32>,
    %scales: tensor<2xf32>, %state: tensor<2x2xf32>) {
  // expected-error @+1 {{coefficient trailing dimension must be statically 5}}
  %output, %next = ondrix.sos_filter_tdf2 %input, %coeffs, %scales, %state {
    numeric = #ondsp.fp<format = f32, contract = fma>
  } : (tensor<4xf32>, tensor<2x?xf32>, tensor<2xf32>, tensor<2x2xf32>)
      -> (tensor<4xf32>, tensor<2x2xf32>)
  return
}

// -----

func.func @rejects_state_layout(
    %input: tensor<4xf32>, %coeffs: tensor<2x5xf32>,
    %scales: tensor<2xf32>, %state: tensor<2x3xf32>) {
  // expected-error @+1 {{state trailing dimension must be statically 2}}
  %output, %next = ondrix.sos_filter_tdf2 %input, %coeffs, %scales, %state {
    numeric = #ondsp.fp<format = f32, contract = off>
  } : (tensor<4xf32>, tensor<2x5xf32>, tensor<2xf32>, tensor<2x3xf32>)
      -> (tensor<4xf32>, tensor<2x3xf32>)
  return
}

// -----

func.func @rejects_empty_cascade(
    %input: tensor<4xf32>, %coeffs: tensor<0x5xf32>,
    %scales: tensor<0xf32>, %state: tensor<0x2xf32>) {
  // expected-error @+1 {{requires at least one second-order section}}
  %output, %next = ondrix.sos_filter_tdf2 %input, %coeffs, %scales, %state {
    numeric = #ondsp.fp<format = f32, contract = off>
  } : (tensor<4xf32>, tensor<0x5xf32>, tensor<0xf32>, tensor<0x2xf32>)
      -> (tensor<4xf32>, tensor<0x2xf32>)
  return
}

// -----

func.func @rejects_section_count_mismatch(
    %input: tensor<4xf32>, %coeffs: tensor<2x5xf32>,
    %scales: tensor<1xf32>, %state: tensor<2x2xf32>) {
  // expected-error @+1 {{coefficient, scale, and state section counts must match}}
  %output, %next = ondrix.sos_filter_tdf2 %input, %coeffs, %scales, %state {
    numeric = #ondsp.fp<format = f32, contract = fma>
  } : (tensor<4xf32>, tensor<2x5xf32>, tensor<1xf32>, tensor<2x2xf32>)
      -> (tensor<4xf32>, tensor<2x2xf32>)
  return
}

// -----

func.func @rejects_fixed_numeric(
    %input: tensor<4xi16>, %coeffs: tensor<2x5xi16>,
    %scales: tensor<2xi16>, %state: tensor<2x2xi16>) {
  // expected-error @+1 {{currently requires an f32 numeric policy}}
  %output, %next = ondrix.sos_filter_tdf2 %input, %coeffs, %scales, %state {
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>
  } : (tensor<4xi16>, tensor<2x5xi16>, tensor<2xi16>, tensor<2x2xi16>)
      -> (tensor<4xi16>, tensor<2x2xi16>)
  return
}

// -----

func.func @rejects_f64(
    %input: tensor<4xf64>, %coeffs: tensor<2x5xf64>,
    %scales: tensor<2xf64>, %state: tensor<2x2xf64>) {
  // expected-error @+1 {{currently requires an f32 numeric policy}}
  %output, %next = ondrix.sos_filter_tdf2 %input, %coeffs, %scales, %state {
    numeric = #ondsp.fp<format = f64, contract = fma>
  } : (tensor<4xf64>, tensor<2x5xf64>, tensor<2xf64>, tensor<2x2xf64>)
      -> (tensor<4xf64>, tensor<2x2xf64>)
  return
}

// -----

func.func @rejects_element_mismatch(
    %input: tensor<4xf32>, %coeffs: tensor<2x5xf32>,
    %scales: tensor<2xf16>, %state: tensor<2x2xf32>) {
  // expected-error @+1 {{input, coefficients, scales, state, and results must use f32}}
  %output, %next = ondrix.sos_filter_tdf2 %input, %coeffs, %scales, %state {
    numeric = #ondsp.fp<format = f32, contract = fma>
  } : (tensor<4xf32>, tensor<2x5xf32>, tensor<2xf16>, tensor<2x2xf32>)
      -> (tensor<4xf32>, tensor<2x2xf32>)
  return
}
