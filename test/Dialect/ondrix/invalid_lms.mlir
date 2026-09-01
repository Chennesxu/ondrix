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
  // expected-error @+1 {{floating-point lms must not specify a raw Q1.15 step size}}
  %0, %1 = ondrix.lms %input, %desired, %weights {
    step_size = 1024,
    fp_step_size = 6.250000e-02 : f32,
    numeric = #ondsp.fp<format = f32, contract = off>
  } : (tensor<8xf32>, tensor<8xf32>, tensor<2xf32>) -> (tensor<8xf32>, tensor<2xf32>)
  return
}

// -----

func.func @rejects_fp_step_size_on_fixed_lms(
    %input: tensor<8xi16>, %desired: tensor<8xi16>, %weights: tensor<2xi16>) {
  // expected-error @+1 {{fixed lms must not specify a floating-point step size}}
  %0, %1 = ondrix.lms %input, %desired, %weights {
    step_size = 1024,
    fp_step_size = 6.250000e-02 : f32,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<8xi16>, tensor<8xi16>, tensor<2xi16>) -> (tensor<8xi16>, tensor<2xi16>)
  return
}

// -----

// The fixed profile admits only the non-negative raw Q1.15 range.
func.func @rejects_negative_fp_step_size(
    %input: tensor<8xf32>, %desired: tensor<8xf32>, %weights: tensor<2xf32>) {
  // expected-error @+1 {{floating-point lms step size must not be negative}}
  %0, %1 = ondrix.lms %input, %desired, %weights {
    fp_step_size = -6.250000e-02 : f32,
    numeric = #ondsp.fp<format = f32, contract = off>
  } : (tensor<8xf32>, tensor<8xf32>, tensor<2xf32>) -> (tensor<8xf32>, tensor<2xf32>)
  return
}

// -----

func.func @rejects_f64_lms(
    %input: tensor<8xf64>, %desired: tensor<8xf64>, %weights: tensor<2xf64>) {
  // expected-error @+1 {{executable lms supports the f32 floating-point format}}
  %0, %1 = ondrix.lms %input, %desired, %weights {
    fp_step_size = 6.250000e-02 : f32,
    numeric = #ondsp.fp<format = f64, contract = off>
  } : (tensor<8xf64>, tensor<8xf64>, tensor<2xf64>) -> (tensor<8xf64>, tensor<2xf64>)
  return
}

// -----

func.func @q31_lms_without_product_rounding(%x: tensor<8xi32>, %d: tensor<8xi32>,
                                            %w: tensor<32xi32>)
    -> (tensor<8xi32>, tensor<32xi32>) {
  // expected-error @below {{requantizes each product by 5 and must declare product_rounding}}
  %e, %a = ondrix.lms %x, %d, %w {
    numeric = #ondsp.fixed<signed, storage = i32, frac = 31>,
    step_size = 1024 : i64, rounding = #ondsp.rounding<nearest_even>
  } : (tensor<8xi32>, tensor<8xi32>, tensor<32xi32>) -> (tensor<8xi32>, tensor<32xi32>)
  return %e, %a : tensor<8xi32>, tensor<32xi32>
}

// -----

func.func @q15_lms_with_product_rounding(%x: tensor<8xi16>, %d: tensor<8xi16>, %w: tensor<32xi16>)
    -> (tensor<8xi16>, tensor<32xi16>) {
  // expected-error @below {{has no product boundary to round}}
  %e, %a = ondrix.lms %x, %d, %w {
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    step_size = 1024 : i64,
    product_rounding = #ondsp.rounding<nearest_even>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<8xi16>, tensor<8xi16>, tensor<32xi16>) -> (tensor<8xi16>, tensor<32xi16>)
  return %e, %a : tensor<8xi16>, tensor<32xi16>
}

// -----

func.func @q31_lms_step_size_out_of_range(%x: tensor<8xi32>, %d: tensor<8xi32>, %w: tensor<4xi32>)
    -> (tensor<8xi32>, tensor<4xi32>) {
  // expected-error @below {{lms step size must be a raw signed Q1.31 value in [0, 2147483647]}}
  %e, %a = ondrix.lms %x, %d, %w {
    numeric = #ondsp.fixed<signed, storage = i32, frac = 31>,
    step_size = 2147483648 : i64,
    product_rounding = #ondsp.rounding<nearest_even>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<8xi32>, tensor<8xi32>, tensor<4xi32>) -> (tensor<8xi32>, tensor<4xi32>)
  return %e, %a : tensor<8xi32>, tensor<4xi32>
}
