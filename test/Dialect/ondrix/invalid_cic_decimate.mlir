// RUN: ondrix-opt %s --split-input-file --verify-diagnostics

func.func @cic_rate_not_power_of_two(%input: tensor<24xi16>) -> tensor<8xi16> {
  // expected-error @below {{cic decimation requires a power-of-two rate in [2, 4096]}}
  %result = ondrix.cic_decimate %input {
    stages = 2 : i64,
    rate = 3 : i64,
    delay = 1 : i64,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    overflow = #ondsp.overflow<wrap>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<24xi16>) -> tensor<8xi16>
  return %result : tensor<8xi16>
}

// -----

func.func @cic_delay_out_of_range(%input: tensor<32xi16>) -> tensor<8xi16> {
  // expected-error @below {{cic decimation requires a differential delay of 1 or 2}}
  %result = ondrix.cic_decimate %input {
    stages = 2 : i64,
    rate = 4 : i64,
    delay = 3 : i64,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    overflow = #ondsp.overflow<wrap>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<32xi16>) -> tensor<8xi16>
  return %result : tensor<8xi16>
}

// -----

func.func @cic_stages_out_of_range(%input: tensor<32xi16>) -> tensor<8xi16> {
  // expected-error @below {{cic decimation requires stages in [1, 8]}}
  %result = ondrix.cic_decimate %input {
    stages = 9 : i64,
    rate = 4 : i64,
    delay = 1 : i64,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    overflow = #ondsp.overflow<wrap>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<32xi16>) -> tensor<8xi16>
  return %result : tensor<8xi16>
}

// -----

// Each attribute is individually admissible; the register width they imply
// is not, and the bound is on the product rather than any one of them.
func.func @cic_growth_exceeds_carrier(%input: tensor<4096xi16>) -> tensor<1xi16> {
  // expected-error @below {{cic decimation requires stages * log2(rate * delay) <= 48}}
  %result = ondrix.cic_decimate %input {
    stages = 5 : i64,
    rate = 4096 : i64,
    delay = 1 : i64,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    overflow = #ondsp.overflow<wrap>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<4096xi16>) -> tensor<1xi16>
  return %result : tensor<1xi16>
}

// -----

// The differential delay is part of the growth, so a rate that fits alone
// can still overflow the carrier once M doubles it.
func.func @cic_delay_pushes_growth_past_carrier(%input: tensor<4096xi16>) -> tensor<1xi16> {
  // expected-error @below {{cic decimation requires stages * log2(rate * delay) <= 48}}
  %result = ondrix.cic_decimate %input {
    stages = 4 : i64,
    rate = 4096 : i64,
    delay = 2 : i64,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    overflow = #ondsp.overflow<wrap>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<4096xi16>) -> tensor<1xi16>
  return %result : tensor<1xi16>
}

// -----

func.func @cic_input_not_a_rate_multiple(%input: tensor<30xi16>) -> tensor<8xi16> {
  // expected-error @below {{executable cic decimation requires static tensor<(R*L)xi16> input}}
  %result = ondrix.cic_decimate %input {
    stages = 2 : i64,
    rate = 4 : i64,
    delay = 1 : i64,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    overflow = #ondsp.overflow<wrap>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<30xi16>) -> tensor<8xi16>
  return %result : tensor<8xi16>
}

// -----

func.func @cic_dynamic_input(%input: tensor<?xi16>) -> tensor<8xi16> {
  // expected-error @below {{executable cic decimation requires static tensor<(R*L)xi16> input}}
  %result = ondrix.cic_decimate %input {
    stages = 2 : i64,
    rate = 4 : i64,
    delay = 1 : i64,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    overflow = #ondsp.overflow<wrap>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<?xi16>) -> tensor<8xi16>
  return %result : tensor<8xi16>
}

// -----

func.func @cic_directed_rounding(%input: tensor<32xi16>) -> tensor<8xi16> {
  // expected-error @below {{cic decimation requires nearest_even or nearest_ties_positive rounding}}
  %result = ondrix.cic_decimate %input {
    stages = 2 : i64,
    rate = 4 : i64,
    delay = 1 : i64,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    overflow = #ondsp.overflow<wrap>,
    rounding = #ondsp.rounding<toward_zero>
  } : (tensor<32xi16>) -> tensor<8xi16>
  return %result : tensor<8xi16>
}

// -----

func.func @cic_rejects_f32(%input: tensor<32xf32>) -> tensor<8xf32> {
  // expected-error @below {{numeric requires #ondsp.fixed<signed, storage = i16, frac = 15>}}
  %result = ondrix.cic_decimate %input {
    stages = 2 : i64,
    rate = 4 : i64,
    delay = 1 : i64,
    numeric = #ondsp.fp<format = f32, contract = off>,
    overflow = #ondsp.overflow<wrap>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<32xf32>) -> tensor<8xf32>
  return %result : tensor<8xf32>
}
