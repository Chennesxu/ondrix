// RUN: ondrix-opt %s --split-input-file --verify-diagnostics

func.func @dct_raw_high_q15(%input: tensor<8xi16>) -> tensor<8xi16> {
  // expected-error @below {{DCT product admits only #ondsp.product<high_raw>, at Q31; the full product is the default}}
  %result = ondrix.dct %input {
    input_numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    output_numeric = #ondsp.fixed<signed, storage = i16, frac = 11>,
    product = #ondsp.product<high_raw>
  } : (tensor<8xi16>) -> tensor<8xi16>
  return %result : tensor<8xi16>
}

// -----

func.func @dct_raw_high_rounds_products(%input: tensor<8xi32>) -> tensor<8xi32> {
  // expected-error @below {{a raw-high DCT has no product rounding to declare: the high half is a floor}}
  %result = ondrix.dct %input {
    input_numeric = #ondsp.fixed<signed, storage = i32, frac = 31>,
    output_numeric = #ondsp.fixed<signed, storage = i32, frac = 27>,
    product = #ondsp.product<high_raw>,
    product_rounding = #ondsp.rounding<nearest_even>
  } : (tensor<8xi32>) -> tensor<8xi32>
  return %result : tensor<8xi32>
}
