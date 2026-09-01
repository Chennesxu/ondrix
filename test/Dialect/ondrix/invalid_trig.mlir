// RUN: ondrix-opt %s --split-input-file --verify-diagnostics

func.func @sine_wrong_rounding(%phase: tensor<8xi16>) -> tensor<8xi16> {
  // expected-error @below {{trigonometric operations require nearest_even rounding}}
  %result = ondrix.sine %phase {
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    rounding = #ondsp.rounding<toward_negative>
  } : (tensor<8xi16>) -> tensor<8xi16>
  return %result : tensor<8xi16>
}

// -----

func.func @cosine_extent_mismatch(%phase: tensor<8xi16>) -> tensor<7xi16> {
  // expected-error @below {{executable trigonometric operations require matching static tensor<Nxi16> input and result with N in [1, 4096]}}
  %result = ondrix.cosine %phase {
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<8xi16>) -> tensor<7xi16>
  return %result : tensor<7xi16>
}

// -----

func.func @sine_oversized(%phase: tensor<8192xi16>) -> tensor<8192xi16> {
  // expected-error @below {{executable trigonometric operations require matching static tensor<Nxi16> input and result with N in [1, 4096]}}
  %result = ondrix.sine %phase {
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<8192xi16>) -> tensor<8192xi16>
  return %result : tensor<8192xi16>
}

// -----

// The phase and the value carry one width, so a Q31 numeric over i16 phases
// fails on the element type rather than being read as a wider value over a
// narrower angle.
func.func @sine_q31_numeric_narrow_phase(%phase: tensor<8xi16>) -> tensor<8xi16> {
  // expected-error @below {{executable trigonometric operations require matching static tensor<Nxi32> input and result with N in [1, 4096]}}
  %result = ondrix.sine %phase {
    numeric = #ondsp.fixed<signed, storage = i32, frac = 31>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<8xi16>) -> tensor<8xi16>
  return %result : tensor<8xi16>
}

// -----

func.func @cosine_non_uniform_q31(%phase: tensor<8xi32>) -> tensor<8xi32> {
  // expected-error @below {{numeric requires #ondsp.fixed<signed, storage = i16, frac = 15> or #ondsp.fixed<signed, storage = i32, frac = 31>}}
  %result = ondrix.cosine %phase {
    numeric = #ondsp.fixed<signed, storage = i32, frac = 28>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<8xi32>) -> tensor<8xi32>
  return %result : tensor<8xi32>
}
