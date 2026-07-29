// RUN: ondrix-opt %s --split-input-file --verify-diagnostics

func.func @rms_odd_extent(%input: tensor<63xi16>) -> tensor<1xi16> {
  // expected-error @below {{executable rms requires static tensor<Nxi16> input with power-of-two N in [2, 4096] and tensor<1xi16> result}}
  %result = ondrix.rms %input {
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<63xi16>) -> tensor<1xi16>
  return %result : tensor<1xi16>
}

// -----

func.func @rms_wrong_result(%input: tensor<64xi16>) -> tensor<2xi16> {
  // expected-error @below {{executable rms requires static tensor<Nxi16> input with power-of-two N in [2, 4096] and tensor<1xi16> result}}
  %result = ondrix.rms %input {
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<64xi16>) -> tensor<2xi16>
  return %result : tensor<2xi16>
}

// -----

// Declaring a fourth rounding mode does not widen an operation that pinned
// its tie rules: rms admits exactly the two modes its lowering and object
// evidence cover, so nearest_ties_positive is refused at the verifier.
func.func @rms_ties_positive_rounding(%input: tensor<64xi16>) -> tensor<1xi16> {
  // expected-error @below {{rms supports toward_negative or nearest_even rounding}}
  %result = ondrix.rms %input {
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    rounding = #ondsp.rounding<nearest_ties_positive>
  } : (tensor<64xi16>) -> tensor<1xi16>
  return %result : tensor<1xi16>
}

// -----

func.func @rms_undersized(%input: tensor<1xi16>) -> tensor<1xi16> {
  // expected-error @below {{executable rms requires static tensor<Nxi16> input with power-of-two N in [2, 4096] and tensor<1xi16> result}}
  %result = ondrix.rms %input {
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<1xi16>) -> tensor<1xi16>
  return %result : tensor<1xi16>
}
