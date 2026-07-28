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
