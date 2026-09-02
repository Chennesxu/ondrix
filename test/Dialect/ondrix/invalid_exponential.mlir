// RUN: ondrix-opt %s --split-input-file --verify-diagnostics

// The two readings are opposite ends of one pair, so neither operation may
// declare the other's.
func.func @log2_with_the_exponential_reading(%a: tensor<8xi16>) -> tensor<8xi16> {
  // expected-error @below {{requires an unsigned Q0.16 input and a signed Q5.11 result}}
  %0 = ondrix.log2 %a {
    numeric = #ondsp.fixed<signed, storage = i16, frac = 11>,
    output_numeric = #ondsp.fixed<unsigned, storage = i16, frac = 16>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<8xi16>) -> tensor<8xi16>
  return %0 : tensor<8xi16>
}

// -----

func.func @exp2_with_the_logarithm_reading(%a: tensor<8xi16>) -> tensor<8xi16> {
  // expected-error @below {{requires a signed Q5.11 input and an unsigned Q0.16 result}}
  %0 = ondrix.exp2 %a {
    numeric = #ondsp.fixed<unsigned, storage = i16, frac = 16>,
    output_numeric = #ondsp.fixed<signed, storage = i16, frac = 11>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<8xi16>) -> tensor<8xi16>
  return %0 : tensor<8xi16>
}

// -----

// A Q15 reading would silently rescale the result by a factor of sixteen.
func.func @log2_with_the_q15_reading(%a: tensor<8xi16>) -> tensor<8xi16> {
  // expected-error @below {{requires an unsigned Q0.16 input and a signed Q5.11 result}}
  %0 = ondrix.log2 %a {
    numeric = #ondsp.fixed<unsigned, storage = i16, frac = 16>,
    output_numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<8xi16>) -> tensor<8xi16>
  return %0 : tensor<8xi16>
}

// -----

func.func @log2_with_a_directed_rule(%a: tensor<8xi16>) -> tensor<8xi16> {
  // expected-error @below {{exponential operations require nearest_even rounding}}
  %0 = ondrix.log2 %a {
    numeric = #ondsp.fixed<unsigned, storage = i16, frac = 16>,
    output_numeric = #ondsp.fixed<signed, storage = i16, frac = 11>,
    rounding = #ondsp.rounding<toward_negative>
  } : (tensor<8xi16>) -> tensor<8xi16>
  return %0 : tensor<8xi16>
}

// -----

func.func @exp2_with_a_dynamic_extent(%a: tensor<?xi16>) -> tensor<?xi16> {
  // expected-error @below {{executable exponential operations require matching static tensor<Nxi16> input and result}}
  %0 = ondrix.exp2 %a {
    numeric = #ondsp.fixed<signed, storage = i16, frac = 11>,
    output_numeric = #ondsp.fixed<unsigned, storage = i16, frac = 16>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<?xi16>) -> tensor<?xi16>
  return %0 : tensor<?xi16>
}

// -----

// The two readings widen together: a Q0.32 magnitude with the Q5.11 exponent
// is neither pair.
func.func @log2_with_mixed_widths(%a: tensor<8xi32>) -> tensor<8xi16> {
  // expected-error @below {{or an unsigned Q0.32 input and a signed Q6.26 result}}
  %0 = ondrix.log2 %a {
    numeric = #ondsp.fixed<unsigned, storage = i32, frac = 32>,
    output_numeric = #ondsp.fixed<signed, storage = i16, frac = 11>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<8xi32>) -> tensor<8xi16>
  return %0 : tensor<8xi16>
}

// -----

func.func @exp2_q31_with_narrow_tensors(%a: tensor<8xi16>) -> tensor<8xi16> {
  // expected-error @below {{executable exponential operations require matching static tensor<Nxi32> input and result}}
  %0 = ondrix.exp2 %a {
    numeric = #ondsp.fixed<signed, storage = i32, frac = 26>,
    output_numeric = #ondsp.fixed<unsigned, storage = i32, frac = 32>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<8xi16>) -> tensor<8xi16>
  return %0 : tensor<8xi16>
}
