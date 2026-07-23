// RUN: ondrix-opt %s -split-input-file -verify-diagnostics

func.func @requires_rank_one(
    %input: tensor<2x4xi16>, %kernel: tensor<3xi16>, %init: tensor<6xi16>) {
  // expected-error @+1 {{requires rank-1 input, kernel, and init tensors}}
  %0 = ondrix.conv1d %input, %kernel, %init {
    accumulator = !ondsp.acc<storage = i40, frac = 30, signed,
                              update_overflow = saturate>,
    dst = #ondsp.fixed<signed, storage = i16, frac = 15>,
    mode = #ondrix.conv1d_mode<convolution>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    overflow = #ondsp.overflow<saturate>,
    product = #ondsp.product<full>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<2x4xi16>, tensor<3xi16>, tensor<6xi16>) -> tensor<6xi16>
  return
}

// -----

func.func @requires_nonempty_kernel(
    %input: tensor<8xi16>, %kernel: tensor<0xi16>, %init: tensor<9xi16>) {
  // expected-error @+1 {{requires at least one kernel element}}
  %0 = ondrix.conv1d %input, %kernel, %init {
    accumulator = !ondsp.acc<storage = i40, frac = 30, signed,
                              update_overflow = saturate>,
    dst = #ondsp.fixed<signed, storage = i16, frac = 15>,
    mode = #ondrix.conv1d_mode<correlation>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    overflow = #ondsp.overflow<saturate>,
    product = #ondsp.product<full>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<8xi16>, tensor<0xi16>, tensor<9xi16>) -> tensor<9xi16>
  return
}

// -----

func.func @requires_exact_result_extent(
    %input: tensor<8xf32>, %kernel: tensor<3xf32>, %init: tensor<5xf32>) {
  // expected-error @+1 {{result length must be 6}}
  %0 = ondrix.conv1d %input, %kernel, %init {
    mode = #ondrix.conv1d_mode<correlation>,
    numeric = #ondsp.fp<format = f32, contract = off>
  } : (tensor<8xf32>, tensor<3xf32>, tensor<5xf32>) -> tensor<5xf32>
  return
}

// -----

func.func @rejects_fixed_policy_on_f32(
    %input: tensor<8xf32>, %kernel: tensor<3xf32>, %init: tensor<6xf32>) {
  // expected-error @+1 {{floating-point conv1d must not specify fixed-point accumulator or export policy}}
  %0 = ondrix.conv1d %input, %kernel, %init {
    accumulator = !ondsp.acc<storage = i40, frac = 30, signed,
                              update_overflow = saturate>,
    mode = #ondrix.conv1d_mode<correlation>,
    numeric = #ondsp.fp<format = f32, contract = off>
  } : (tensor<8xf32>, tensor<3xf32>, tensor<6xf32>) -> tensor<6xf32>
  return
}
