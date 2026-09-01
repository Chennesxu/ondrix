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

// -----

func.func @fp_rms_rounding(%input: tensor<8xf32>) -> tensor<1xf32> {
  // expected-error @below {{floating-point rms rounds at no declared boundary of its own}}
  %result = ondrix.rms %input {
    numeric = #ondsp.fp<format = f32, contract = off>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<8xf32>) -> tensor<1xf32>
  return %result : tensor<1xf32>
}

// -----

func.func @fixed_rms_without_rounding(%input: tensor<8xi16>) -> tensor<1xi16> {
  // expected-error @below {{rms supports toward_negative or nearest_even rounding}}
  %result = ondrix.rms %input {
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>
  } : (tensor<8xi16>) -> tensor<1xi16>
  return %result : tensor<1xi16>
}

// -----

func.func @q31_rms_without_input_rounding(%input: tensor<4096xi32>) -> tensor<1xi32> {
  // expected-error @below {{requantizes each input by 6 before squaring and must declare input_rounding}}
  %result = ondrix.rms %input {
    numeric = #ondsp.fixed<signed, storage = i32, frac = 31>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<4096xi32>) -> tensor<1xi32>
  return %result : tensor<1xi32>
}

// -----

func.func @q15_rms_with_input_rounding(%input: tensor<64xi16>) -> tensor<1xi16> {
  // expected-error @below {{has no pre-shift boundary to round}}
  %result = ondrix.rms %input {
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    input_rounding = #ondsp.rounding<nearest_even>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<64xi16>) -> tensor<1xi16>
  return %result : tensor<1xi16>
}

// -----

func.func @f32_rms_with_input_rounding(%input: tensor<10xf32>) -> tensor<1xf32> {
  // expected-error @below {{floating-point rms has no pre-shift boundary to round}}
  %result = ondrix.rms %input {
    numeric = #ondsp.fp<format = f32, contract = off>,
    input_rounding = #ondsp.rounding<nearest_even>
  } : (tensor<10xf32>) -> tensor<1xf32>
  return %result : tensor<1xf32>
}
