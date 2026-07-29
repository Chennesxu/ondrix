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
