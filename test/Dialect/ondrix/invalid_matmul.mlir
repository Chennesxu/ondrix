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
  // expected-error @below {{matmul requires nearest_even rounding}}
  %c = ondrix.matmul %a, %b {
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    rounding = #ondsp.rounding<toward_negative>
  } : (tensor<4x8xi16>, tensor<8x3xi16>) -> tensor<4x3xi16>
  return %c : tensor<4x3xi16>
}
