// RUN: ondrix-opt %s --split-input-file --verify-diagnostics

func.func @gain_out_of_range(%input: tensor<8xi16>) -> tensor<8xi16> {
  // expected-error @below {{gain constant must be a raw signed Q1.15 value in [-32768, 32767]}}
  %result = ondrix.gain %input {
    gain = 32768 : i64,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<8xi16>) -> tensor<8xi16>
  return %result : tensor<8xi16>
}

// -----

func.func @gain_wrong_rounding(%input: tensor<8xi16>) -> tensor<8xi16> {
  // expected-error @below {{gain requires nearest_even rounding}}
  %result = ondrix.gain %input {
    gain = 16384 : i64,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    rounding = #ondsp.rounding<toward_negative>
  } : (tensor<8xi16>) -> tensor<8xi16>
  return %result : tensor<8xi16>
}

// -----

func.func @gain_dynamic_shape(%input: tensor<?xi16>) -> tensor<?xi16> {
  // expected-error @below {{executable gain requires matching static tensor<Nxi16> input and result with N in [1, 4096]}}
  %result = ondrix.gain %input {
    gain = 16384 : i64,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<?xi16>) -> tensor<?xi16>
  return %result : tensor<?xi16>
}

// -----

func.func @gain_extent_mismatch(%input: tensor<8xi16>) -> tensor<7xi16> {
  // expected-error @below {{executable gain requires matching static tensor<Nxi16> input and result with N in [1, 4096]}}
  %result = ondrix.gain %input {
    gain = 16384 : i64,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<8xi16>) -> tensor<7xi16>
  return %result : tensor<7xi16>
}

// -----

func.func @gain_oversized(%input: tensor<8192xi16>) -> tensor<8192xi16> {
  // expected-error @below {{executable gain requires matching static tensor<Nxi16> input and result with N in [1, 4096]}}
  %result = ondrix.gain %input {
    gain = 16384 : i64,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<8192xi16>) -> tensor<8192xi16>
  return %result : tensor<8192xi16>
}

// -----

func.func @gain_wrong_numeric(%input: tensor<8xi16>) -> tensor<8xi16> {
  // expected-error @below {{numeric}}
  %result = ondrix.gain %input {
    gain = 16384 : i64,
    numeric = #ondsp.fixed<signed, storage = i32, frac = 31>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<8xi16>) -> tensor<8xi16>
  return %result : tensor<8xi16>
}
