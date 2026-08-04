// RUN: ondrix-opt %s --split-input-file --verify-diagnostics

// A binary member must check both sides, not only the one whose type the
// result happens to match.
func.func @add_mismatched_right_extent(%a: tensor<8xi16>, %b: tensor<4xi16>) -> tensor<8xi16> {
  // expected-error @below {{executable elementwise operations require matching static tensor<Nxi16> operands and result}}
  %0 = ondrix.add %a, %b {
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    overflow = #ondsp.overflow<saturate>
  } : (tensor<8xi16>, tensor<4xi16>) -> tensor<8xi16>
  return %0 : tensor<8xi16>
}

// -----

func.func @sub_dynamic_extent(%a: tensor<?xi16>, %b: tensor<?xi16>) -> tensor<?xi16> {
  // expected-error @below {{executable elementwise operations require matching static tensor<Nxi16> operands and result}}
  %0 = ondrix.sub %a, %b {
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    overflow = #ondsp.overflow<saturate>
  } : (tensor<?xi16>, tensor<?xi16>) -> tensor<?xi16>
  return %0 : tensor<?xi16>
}

// -----

func.func @mult_rejects_f32(%a: tensor<8xf32>, %b: tensor<8xf32>) -> tensor<8xf32> {
  // expected-error @below {{numeric requires #ondsp.fixed<signed, storage = i16, frac = 15>}}
  %0 = ondrix.mult %a, %b {
    numeric = #ondsp.fp<format = f32, contract = off>,
    rounding = #ondsp.rounding<nearest_even>,
    overflow = #ondsp.overflow<saturate>
  } : (tensor<8xf32>, tensor<8xf32>) -> tensor<8xf32>
  return %0 : tensor<8xf32>
}

// -----

func.func @offset_bias_out_of_range(%a: tensor<8xi16>) -> tensor<8xi16> {
  // expected-error @below {{offset bias must be a raw signed Q1.15 value in [-32768, 32767]}}
  %0 = ondrix.offset %a {
    bias = 32768 : i64,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    overflow = #ondsp.overflow<saturate>
  } : (tensor<8xi16>) -> tensor<8xi16>
  return %0 : tensor<8xi16>
}

// -----

func.func @shift_amount_out_of_range(%a: tensor<8xi16>) -> tensor<8xi16> {
  // expected-error @below {{shift amount must lie in [-15, 15]}}
  %0 = ondrix.shift %a {
    amount = 16 : i64,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    rounding = #ondsp.rounding<nearest_even>,
    overflow = #ondsp.overflow<saturate>
  } : (tensor<8xi16>) -> tensor<8xi16>
  return %0 : tensor<8xi16>
}

// -----

func.func @abs_extent_out_of_range(%a: tensor<4097xi16>) -> tensor<4097xi16> {
  // expected-error @below {{executable elementwise operations require matching static tensor<Nxi16> operands and result}}
  %0 = ondrix.abs %a {
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    overflow = #ondsp.overflow<saturate>
  } : (tensor<4097xi16>) -> tensor<4097xi16>
  return %0 : tensor<4097xi16>
}

// -----

func.func @negate_rejects_memref(%a: memref<8xi16>) -> tensor<8xi16> {
  // expected-error @below {{operand #0 must be ranked tensor of any type values}}
  %0 = "ondrix.negate"(%a) {
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    overflow = #ondsp.overflow<saturate>
  } : (memref<8xi16>) -> tensor<8xi16>
  return %0 : tensor<8xi16>
}
