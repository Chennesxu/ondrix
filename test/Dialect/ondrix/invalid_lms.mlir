// RUN: ondrix-opt %s --split-input-file --verify-diagnostics

func.func @lms_negative_step(%x: tensor<8xi16>, %d: tensor<8xi16>, %w: tensor<4xi16>)
    -> (tensor<8xi16>, tensor<4xi16>) {
  // expected-error @below {{lms step size must be a raw signed Q1.15 value in [0, 32767]}}
  %e, %wf = ondrix.lms %x, %d, %w {
    step_size = -1 : i64,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<8xi16>, tensor<8xi16>, tensor<4xi16>) -> (tensor<8xi16>, tensor<4xi16>)
  return %e, %wf : tensor<8xi16>, tensor<4xi16>
}

// -----

func.func @lms_wrong_rounding(%x: tensor<8xi16>, %d: tensor<8xi16>, %w: tensor<4xi16>)
    -> (tensor<8xi16>, tensor<4xi16>) {
  // expected-error @below {{lms requires nearest_even rounding}}
  %e, %wf = ondrix.lms %x, %d, %w {
    step_size = 4096 : i64,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    rounding = #ondsp.rounding<toward_negative>
  } : (tensor<8xi16>, tensor<8xi16>, tensor<4xi16>) -> (tensor<8xi16>, tensor<4xi16>)
  return %e, %wf : tensor<8xi16>, tensor<4xi16>
}

// -----

func.func @lms_desired_mismatch(%x: tensor<8xi16>, %d: tensor<7xi16>, %w: tensor<4xi16>)
    -> (tensor<8xi16>, tensor<4xi16>) {
  // expected-error @below {{executable lms requires matching static tensor<Nxi16> input, desired, and error with N in [1, 4096]}}
  %e, %wf = ondrix.lms %x, %d, %w {
    step_size = 4096 : i64,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<8xi16>, tensor<7xi16>, tensor<4xi16>) -> (tensor<8xi16>, tensor<4xi16>)
  return %e, %wf : tensor<8xi16>, tensor<4xi16>
}

// -----

func.func @lms_oversized_taps(%x: tensor<8xi16>, %d: tensor<8xi16>, %w: tensor<128xi16>)
    -> (tensor<8xi16>, tensor<128xi16>) {
  // expected-error @below {{executable lms requires matching static tensor<Kxi16> weights and adapted weights with K in [1, 64]}}
  %e, %wf = ondrix.lms %x, %d, %w {
    step_size = 4096 : i64,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<8xi16>, tensor<8xi16>, tensor<128xi16>) -> (tensor<8xi16>, tensor<128xi16>)
  return %e, %wf : tensor<8xi16>, tensor<128xi16>
}

// -----

func.func @lms_dynamic_input(%x: tensor<?xi16>, %d: tensor<?xi16>, %w: tensor<4xi16>)
    -> (tensor<?xi16>, tensor<4xi16>) {
  // expected-error @below {{executable lms requires matching static tensor<Nxi16> input, desired, and error with N in [1, 4096]}}
  %e, %wf = ondrix.lms %x, %d, %w {
    step_size = 4096 : i64,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<?xi16>, tensor<?xi16>, tensor<4xi16>) -> (tensor<?xi16>, tensor<4xi16>)
  return %e, %wf : tensor<?xi16>, tensor<4xi16>
}

// -----

func.func @rejects_missing_fp_step_size(
    %input: tensor<8xf32>, %desired: tensor<8xf32>, %weights: tensor<2xf32>) {
  // expected-error @+1 {{floating-point lms requires the fp_step_size constant}}
  %0, %1 = ondrix.lms %input, %desired, %weights {
    numeric = #ondsp.fp<format = f32, contract = off>
  } : (tensor<8xf32>, tensor<8xf32>, tensor<2xf32>) -> (tensor<8xf32>, tensor<2xf32>)
  return
}

// -----

func.func @rejects_fixed_step_size_on_fp_lms(
    %input: tensor<8xf32>, %desired: tensor<8xf32>, %weights: tensor<2xf32>) {
  // expected-error @+1 {{floating-point lms rounds at no declared boundary of its own}}
  %0, %1 = ondrix.lms %input, %desired, %weights {
    step_size = 1024,
    fp_step_size = 6.250000e-02 : f32,
    numeric = #ondsp.fp<format = f32, contract = off>
  } : (tensor<8xf32>, tensor<8xf32>, tensor<2xf32>) -> (tensor<8xf32>, tensor<2xf32>)
  return
}
