// RUN: ondrix-opt %s --split-input-file --verify-diagnostics

func.func @matmul_inner_mismatch(%a: tensor<4x8xi16>, %b: tensor<7x3xi16>) -> tensor<4x3xi16> {
  // expected-error @below {{executable matmul requires static tensor<MxKxi16> x tensor<KxNxi16> -> tensor<MxNxi16> with M, K, N in [1, 64]}}
  %c = ondrix.matmul %a, %b {
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<4x8xi16>, tensor<7x3xi16>) -> tensor<4x3xi16>
  return %c : tensor<4x3xi16>
}

// -----

func.func @matmul_result_mismatch(%a: tensor<4x8xi16>, %b: tensor<8x3xi16>) -> tensor<3x4xi16> {
  // expected-error @below {{executable matmul requires static tensor<MxKxi16> x tensor<KxNxi16> -> tensor<MxNxi16> with M, K, N in [1, 64]}}
  %c = ondrix.matmul %a, %b {
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<4x8xi16>, tensor<8x3xi16>) -> tensor<3x4xi16>
  return %c : tensor<3x4xi16>
}

// -----

func.func @matmul_oversized(%a: tensor<4x128xi16>, %b: tensor<128x3xi16>) -> tensor<4x3xi16> {
  // expected-error @below {{executable matmul requires static tensor<MxKxi16> x tensor<KxNxi16> -> tensor<MxNxi16> with M, K, N in [1, 64]}}
  %c = ondrix.matmul %a, %b {
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<4x128xi16>, tensor<128x3xi16>) -> tensor<4x3xi16>
  return %c : tensor<4x3xi16>
}

// -----

func.func @matmul_rank_one(%a: tensor<8xi16>, %b: tensor<8x3xi16>) -> tensor<3xi16> {
  // expected-error @below {{executable matmul requires static tensor<MxKxi16> x tensor<KxNxi16> -> tensor<MxNxi16> with M, K, N in [1, 64]}}
  %c = ondrix.matmul %a, %b {
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<8xi16>, tensor<8x3xi16>) -> tensor<3xi16>
  return %c : tensor<3xi16>
}

// -----

func.func @matmul_wrong_rounding(%a: tensor<4x8xi16>, %b: tensor<8x3xi16>) -> tensor<4x3xi16> {
  // expected-error @below {{matmul rounding must be nearest_even, toward_negative, or nearest_ties_positive}}
  %c = ondrix.matmul %a, %b {
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    rounding = #ondsp.rounding<toward_zero>
  } : (tensor<4x8xi16>, tensor<8x3xi16>) -> tensor<4x3xi16>
  return %c : tensor<4x3xi16>
}

// -----

func.func @fp_matmul_rounding(%a: tensor<2x2xf32>, %b: tensor<2x2xf32>) -> tensor<2x2xf32> {
  // expected-error @below {{floating-point matmul has no requantization boundary to round}}
  %c = ondrix.matmul %a, %b {
    numeric = #ondsp.fp<format = f32, contract = off>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
  return %c : tensor<2x2xf32>
}

// -----

func.func @fp_matmul_element(%a: tensor<2x2xi16>, %b: tensor<2x2xi16>) -> tensor<2x2xi16> {
  // expected-error @below {{executable matmul requires static tensor<MxKxf32>}}
  %c = ondrix.matmul %a, %b {
    numeric = #ondsp.fp<format = f32, contract = off>
  } : (tensor<2x2xi16>, tensor<2x2xi16>) -> tensor<2x2xi16>
  return %c : tensor<2x2xi16>
}

// -----

func.func @fixed_matmul_without_rounding(%a: tensor<2x2xi16>, %b: tensor<2x2xi16>) -> tensor<2x2xi16> {
  // expected-error @below {{matmul rounding must be nearest_even, toward_negative, or nearest_ties_positive}}
  %c = ondrix.matmul %a, %b {
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>
  } : (tensor<2x2xi16>, tensor<2x2xi16>) -> tensor<2x2xi16>
  return %c : tensor<2x2xi16>
}

// -----

func.func @q31_matmul_without_product_rounding(%a: tensor<2x64xi32>, %b: tensor<64x2xi32>)
    -> tensor<2x2xi32> {
  // expected-error @below {{requantizes each product by 6 and must declare product_rounding}}
  %result = ondrix.matmul %a, %b {
    numeric = #ondsp.fixed<signed, storage = i32, frac = 31>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<2x64xi32>, tensor<64x2xi32>) -> tensor<2x2xi32>
  return %result : tensor<2x2xi32>
}

// -----

func.func @q15_matmul_with_product_rounding(%a: tensor<2x8xi16>, %b: tensor<8x2xi16>)
    -> tensor<2x2xi16> {
  // expected-error @below {{has no product boundary to round}}
  %result = ondrix.matmul %a, %b {
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    product_rounding = #ondsp.rounding<nearest_even>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<2x8xi16>, tensor<8x2xi16>) -> tensor<2x2xi16>
  return %result : tensor<2x2xi16>
}

// -----

func.func @q31_matmul_k1_with_product_rounding(%a: tensor<2x1xi32>, %b: tensor<1x2xi32>)
    -> tensor<2x2xi32> {
  // expected-error @below {{has no product boundary to round}}
  %result = ondrix.matmul %a, %b {
    numeric = #ondsp.fixed<signed, storage = i32, frac = 31>,
    product_rounding = #ondsp.rounding<nearest_even>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<2x1xi32>, tensor<1x2xi32>) -> tensor<2x2xi32>
  return %result : tensor<2x2xi32>
}

// -----

func.func @f32_matmul_with_product_rounding(%a: tensor<2x8xf32>, %b: tensor<8x2xf32>)
    -> tensor<2x2xf32> {
  // expected-error @below {{floating-point matmul has no product boundary to round}}
  %result = ondrix.matmul %a, %b {
    numeric = #ondsp.fp<format = f32, contract = off>,
    product_rounding = #ondsp.rounding<nearest_even>
  } : (tensor<2x8xf32>, tensor<8x2xf32>) -> tensor<2x2xf32>
  return %result : tensor<2x2xf32>
}
