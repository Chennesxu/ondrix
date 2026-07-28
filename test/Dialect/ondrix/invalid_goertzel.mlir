// RUN: ondrix-opt %s --split-input-file --verify-diagnostics

func.func @goertzel_bin_above_nyquist(%input: tensor<64xi16>) -> tensor<1xi64> {
  // expected-error @below {{goertzel bin must lie in [0, N/2]}}
  %energy = ondrix.goertzel %input {
    bin = 33 : i64,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<64xi16>) -> tensor<1xi64>
  return %energy : tensor<1xi64>
}

// -----

func.func @goertzel_negative_bin(%input: tensor<64xi16>) -> tensor<1xi64> {
  // expected-error @below {{goertzel bin must lie in [0, N/2]}}
  %energy = ondrix.goertzel %input {
    bin = -1 : i64,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<64xi16>) -> tensor<1xi64>
  return %energy : tensor<1xi64>
}

// -----

func.func @goertzel_wrong_energy_type(%input: tensor<64xi16>) -> tensor<1xi16> {
  // expected-error @below {{executable goertzel requires static tensor<Nxi16> input with N in [2, 4096] and tensor<1xi64> energy}}
  %energy = ondrix.goertzel %input {
    bin = 5 : i64,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<64xi16>) -> tensor<1xi16>
  return %energy : tensor<1xi16>
}

// -----

func.func @goertzel_wrong_rounding(%input: tensor<64xi16>) -> tensor<1xi64> {
  // expected-error @below {{goertzel requires nearest_even rounding}}
  %energy = ondrix.goertzel %input {
    bin = 5 : i64,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    rounding = #ondsp.rounding<toward_negative>
  } : (tensor<64xi16>) -> tensor<1xi64>
  return %energy : tensor<1xi64>
}
