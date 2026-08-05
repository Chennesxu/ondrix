// RUN: ondrix-opt %s -split-input-file -verify-diagnostics

func.func @rejects_wrong_state_length(
    %input: tensor<4xi16>, %coeffs: tensor<3xi16>, %state: tensor<1xi16>) {
  // expected-error @+1 {{state length must be 2}}
  %output, %next = ondrix.fir_stream %input, %coeffs, %state {
    accumulator = !ondsp.acc<storage = i40, frac = 30, signed,
                              update_overflow = saturate>,
    dst = #ondsp.fixed<signed, storage = i16, frac = 15>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    overflow = #ondsp.overflow<saturate>,
    product = #ondsp.product<full>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<4xi16>, tensor<3xi16>, tensor<1xi16>)
      -> (tensor<4xi16>, tensor<2xi16>)
  return
}

// -----

func.func @rejects_wrong_output_length(
    %input: tensor<4xf32>, %coeffs: tensor<3xf32>, %state: tensor<2xf32>) {
  // expected-error @+1 {{output length must equal input chunk length}}
  %output, %next = ondrix.fir_stream %input, %coeffs, %state {
    numeric = #ondsp.fp<format = f32, contract = fma>
  } : (tensor<4xf32>, tensor<3xf32>, tensor<2xf32>)
      -> (tensor<3xf32>, tensor<2xf32>)
  return
}

// -----

func.func @rejects_wrong_next_state_length(
    %input: tensor<4xf32>, %coeffs: tensor<3xf32>, %state: tensor<2xf32>) {
  // expected-error @+1 {{next-state length must be 2}}
  %output, %next = ondrix.fir_stream %input, %coeffs, %state {
    numeric = #ondsp.fp<format = f32, contract = fma>
  } : (tensor<4xf32>, tensor<3xf32>, tensor<2xf32>)
      -> (tensor<4xf32>, tensor<1xf32>)
  return
}

// -----

func.func @rejects_state_result_mismatch_with_dynamic_coefficients(
    %input: tensor<4xf32>, %coeffs: tensor<?xf32>, %state: tensor<2xf32>) {
  // expected-error @+1 {{next-state length must equal state length}}
  %output, %next = ondrix.fir_stream %input, %coeffs, %state {
    numeric = #ondsp.fp<format = f32, contract = fma>
  } : (tensor<4xf32>, tensor<?xf32>, tensor<2xf32>)
      -> (tensor<4xf32>, tensor<1xf32>)
  return
}

// -----

func.func @rejects_empty_coefficients(
    %input: tensor<4xf32>, %coeffs: tensor<0xf32>, %state: tensor<0xf32>) {
  // expected-error @+1 {{requires at least one coefficient}}
  %output, %next = ondrix.fir_stream %input, %coeffs, %state {
    numeric = #ondsp.fp<format = f32, contract = fma>
  } : (tensor<4xf32>, tensor<0xf32>, tensor<0xf32>)
      -> (tensor<4xf32>, tensor<0xf32>)
  return
}

// -----

func.func @rejects_non_rank_one(
    %input: tensor<2x2xf32>, %coeffs: tensor<3xf32>, %state: tensor<2xf32>) {
  // expected-error @+1 {{requires rank-1 input, coefficients, state, and results}}
  %output, %next = ondrix.fir_stream %input, %coeffs, %state {
    numeric = #ondsp.fp<format = f32, contract = fma>
  } : (tensor<2x2xf32>, tensor<3xf32>, tensor<2xf32>)
      -> (tensor<4xf32>, tensor<2xf32>)
  return
}

// -----

func.func @fixed_requires_lifecycle(
    %input: tensor<4xi16>, %coeffs: tensor<3xi16>, %state: tensor<2xi16>) {
  // expected-error @+1 {{fixed FIR stream requires accumulator, dst, rounding, and overflow attributes}}
  %output, %next = ondrix.fir_stream %input, %coeffs, %state {
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    product = #ondsp.product<full>
  } : (tensor<4xi16>, tensor<3xi16>, tensor<2xi16>)
      -> (tensor<4xi16>, tensor<2xi16>)
  return
}

// -----

func.func @fp_rejects_fixed_lifecycle(
    %input: tensor<4xf32>, %coeffs: tensor<3xf32>, %state: tensor<2xf32>) {
  // expected-error @+1 {{floating-point FIR stream must not specify fixed-point accumulator or export policy}}
  %output, %next = ondrix.fir_stream %input, %coeffs, %state {
    accumulator = !ondsp.acc<storage = i40, frac = 30, signed,
                              update_overflow = saturate>,
    numeric = #ondsp.fp<format = f32, contract = fma>
  } : (tensor<4xf32>, tensor<3xf32>, tensor<2xf32>)
      -> (tensor<4xf32>, tensor<2xf32>)
  return
}

// -----

func.func @rejects_encoded_input(
    %input: tensor<4xf32, "encoded">, %coeffs: tensor<3xf32>, %state: tensor<2xf32>) {
  // expected-error @+1 {{does not support encoded tensor types}}
  %output, %next = ondrix.fir_stream %input, %coeffs, %state {
    numeric = #ondsp.fp<format = f32, contract = fma>
  } : (tensor<4xf32, "encoded">, tensor<3xf32>, tensor<2xf32>)
      -> (tensor<4xf32>, tensor<2xf32>)
  return
}

// -----

func.func @rejects_encoded_result(
    %input: tensor<4xf32>, %coeffs: tensor<3xf32>, %state: tensor<2xf32>) {
  // expected-error @+1 {{does not support encoded tensor types}}
  %output, %next = ondrix.fir_stream %input, %coeffs, %state {
    numeric = #ondsp.fp<format = f32, contract = fma>
  } : (tensor<4xf32>, tensor<3xf32>, tensor<2xf32>)
      -> (tensor<4xf32, "encoded">, tensor<2xf32>)
  return
}

// -----

func.func @fir_stream_rejects_f64(
    %input: tensor<4xf64>, %coeffs: tensor<3xf64>, %state: tensor<2xf64>) {
  // expected-error @+1 {{executable FIR stream supports the f32 floating-point format}}
  %output, %next = ondrix.fir_stream %input, %coeffs, %state {
    numeric = #ondsp.fp<format = f64, contract = fma>
  } : (tensor<4xf64>, tensor<3xf64>, tensor<2xf64>)
      -> (tensor<4xf64>, tensor<2xf64>)
  return
}
