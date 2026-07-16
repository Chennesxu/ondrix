// RUN: ondrix-opt %s -split-input-file -verify-diagnostics

func.func @wrong_full_product_frac(
    %acc: !ondsp.acc<storage = i40, frac = 29, signed, update_overflow = saturate>, %a: i16, %b: i16)
    -> !ondsp.acc<storage = i40, frac = 29, signed, update_overflow = saturate> {
  // expected-error@+1 {{accumulator frac 29 does not match expected frac 30 for full product}}
  %0 = ondsp.mac %acc, %a, %b {numeric = #ondsp.fixed<signed, storage = i16, frac = 15>, product = #ondsp.product<full>} : (!ondsp.acc<storage = i40, frac = 29, signed, update_overflow = saturate>, i16, i16) -> !ondsp.acc<storage = i40, frac = 29, signed, update_overflow = saturate>
  return %0 : !ondsp.acc<storage = i40, frac = 29, signed, update_overflow = saturate>
}

// -----

func.func @fixed_reduce_without_product(%a: memref<8xi16>, %b: memref<8xi16>) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate> {
  %zero = ondsp.acc_zero : !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  // expected-error@+1 {{fixed numeric policy requires a product attribute}}
  %0 = ondsp.reduce_mac %zero, %a, %b {numeric = #ondsp.fixed<signed, storage = i16, frac = 15>} : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>, memref<8xi16>, memref<8xi16>) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  return %0 : !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
}

// -----

func.func @fp_reduce_with_integer_product(%a: memref<8xf32>, %b: memref<8xf32>) -> f32 {
  %zero = arith.constant 0.0 : f32
  // expected-error@+1 {{floating-point numeric policy must not specify a product attribute}}
  %0 = ondsp.reduce_mac %zero, %a, %b {numeric = #ondsp.fp<format = f32, contract = fma>, product = #ondsp.product<full>} : (f32, memref<8xf32>, memref<8xf32>) -> f32
  return %0 : f32
}

// -----

func.func @negative_high_product_frac(
    %acc: !ondsp.acc<storage = i40, frac = 0, signed, update_overflow = saturate>, %a: i16, %b: i16)
    -> !ondsp.acc<storage = i40, frac = 0, signed, update_overflow = saturate> {
  // expected-error@+1 {{raw high product fractional position would be negative}}
  %0 = ondsp.mac %acc, %a, %b {numeric = #ondsp.fixed<signed, storage = i16, frac = 7>, product = #ondsp.product<high_raw>} : (!ondsp.acc<storage = i40, frac = 0, signed, update_overflow = saturate>, i16, i16) -> !ondsp.acc<storage = i40, frac = 0, signed, update_overflow = saturate>
  return %0 : !ondsp.acc<storage = i40, frac = 0, signed, update_overflow = saturate>
}

// -----

func.func @accumulator_signedness_mismatch(
    %acc: !ondsp.acc<storage = i40, frac = 30, unsigned, update_overflow = saturate>, %a: i16, %b: i16)
    -> !ondsp.acc<storage = i40, frac = 30, unsigned, update_overflow = saturate> {
  // expected-error@+1 {{accumulator signedness must match the fixed numeric policy}}
  %0 = ondsp.mac %acc, %a, %b {numeric = #ondsp.fixed<signed, storage = i16, frac = 15>, product = #ondsp.product<full>} : (!ondsp.acc<storage = i40, frac = 30, unsigned, update_overflow = saturate>, i16, i16) -> !ondsp.acc<storage = i40, frac = 30, unsigned, update_overflow = saturate>
  return %0 : !ondsp.acc<storage = i40, frac = 30, unsigned, update_overflow = saturate>
}

// -----

func.func @fixed_reduce_requires_accumulator(%a: memref<8xi16>, %b: memref<8xi16>) -> f32 {
  %zero = arith.constant 0.0 : f32
  // expected-error@+1 {{fixed reduce_mac initial and result must use !ondsp.acc}}
  %0 = ondsp.reduce_mac %zero, %a, %b {numeric = #ondsp.fixed<signed, storage = i16, frac = 15>, product = #ondsp.product<full>} : (f32, memref<8xi16>, memref<8xi16>) -> f32
  return %0 : f32
}

// -----

func.func @fixed_reduce_requires_product_frac(%a: memref<8xi16>, %b: memref<8xi16>) -> !ondsp.acc<storage = i40, frac = 29, signed, update_overflow = saturate> {
  %zero = ondsp.acc_zero : !ondsp.acc<storage = i40, frac = 29, signed, update_overflow = saturate>
  // expected-error@+1 {{accumulator frac 29 does not match expected frac 30 for full product}}
  %0 = ondsp.reduce_mac %zero, %a, %b {numeric = #ondsp.fixed<signed, storage = i16, frac = 15>, product = #ondsp.product<full>} : (!ondsp.acc<storage = i40, frac = 29, signed, update_overflow = saturate>, memref<8xi16>, memref<8xi16>) -> !ondsp.acc<storage = i40, frac = 29, signed, update_overflow = saturate>
  return %0 : !ondsp.acc<storage = i40, frac = 29, signed, update_overflow = saturate>
}

// -----

func.func @reduce_mac_rejects_mixed_domains(
    %lhs: memref<8xf32>, %rhs: f32) -> f32 {
  %zero = arith.constant 0.0 : f32
  // expected-error@+1 {{shaped operands must be rank-1}}
  %0 = ondsp.reduce_mac %zero, %lhs, %rhs {numeric = #ondsp.fp<format = f32, contract = fma>} : (f32, memref<8xf32>, f32) -> f32
  return %0 : f32
}

// -----

func.func @reduce_mac_rejects_unranked(
    %lhs: memref<*xf32>, %rhs: memref<*xf32>) -> f32 {
  %zero = arith.constant 0.0 : f32
  // expected-error@+1 {{shaped operands must be rank-1}}
  %0 = ondsp.reduce_mac %zero, %lhs, %rhs {numeric = #ondsp.fp<format = f32, contract = fma>} : (f32, memref<*xf32>, memref<*xf32>) -> f32
  return %0 : f32
}

// -----

func.func @floating_reduce_mac_requires_scalar_result(
    %lhs: vector<8xf32>, %rhs: vector<8xf32>) -> vector<8xf32> {
  %zero = arith.constant dense<0.0> : vector<8xf32>
  // expected-error@+1 {{floating-point reduce_mac initial and result must match numeric format}}
  %0 = ondsp.reduce_mac %zero, %lhs, %rhs {numeric = #ondsp.fp<format = f32, contract = fma>} : (vector<8xf32>, vector<8xf32>, vector<8xf32>) -> vector<8xf32>
  return %0 : vector<8xf32>
}

// -----

func.func @reduce_mac_rejects_scalable_vectors(
    %lhs: vector<8xf32>, %rhs: vector<[8]xf32>) -> f32 {
  %zero = arith.constant 0.0 : f32
  // expected-error@+1 {{scalable vector operands are not supported}}
  %0 = ondsp.reduce_mac %zero, %lhs, %rhs {numeric = #ondsp.fp<format = f32, contract = fma>} : (f32, vector<8xf32>, vector<[8]xf32>) -> f32
  return %0 : f32
}

// -----

func.func @fixed_reduce_requires_matching_signedness(%a: memref<8xi16>, %b: memref<8xi16>) -> !ondsp.acc<storage = i40, frac = 30, unsigned, update_overflow = saturate> {
  %zero = ondsp.acc_zero : !ondsp.acc<storage = i40, frac = 30, unsigned, update_overflow = saturate>
  // expected-error@+1 {{accumulator signedness must match the fixed numeric policy}}
  %0 = ondsp.reduce_mac %zero, %a, %b {numeric = #ondsp.fixed<signed, storage = i16, frac = 15>, product = #ondsp.product<full>} : (!ondsp.acc<storage = i40, frac = 30, unsigned, update_overflow = saturate>, memref<8xi16>, memref<8xi16>) -> !ondsp.acc<storage = i40, frac = 30, unsigned, update_overflow = saturate>
  return %0 : !ondsp.acc<storage = i40, frac = 30, unsigned, update_overflow = saturate>
}

// -----

func.func @unsigned_full_product_is_undefined(
    %acc: !ondsp.acc<storage = i40, frac = 30, unsigned, update_overflow = saturate>,
    %a: i16, %b: i16)
    -> !ondsp.acc<storage = i40, frac = 30, unsigned, update_overflow = saturate> {
  // expected-error@+1 {{fixed product semantics currently require a signed numeric policy}}
  %0 = ondsp.mac %acc, %a, %b {numeric = #ondsp.fixed<unsigned, storage = i16, frac = 15>, product = #ondsp.product<full>} : (!ondsp.acc<storage = i40, frac = 30, unsigned, update_overflow = saturate>, i16, i16) -> !ondsp.acc<storage = i40, frac = 30, unsigned, update_overflow = saturate>
  return %0 : !ondsp.acc<storage = i40, frac = 30, unsigned, update_overflow = saturate>
}

// -----

func.func @unsigned_high_raw_product_is_undefined(
    %acc: !ondsp.acc<storage = i40, frac = 30, unsigned, update_overflow = saturate>,
    %a: i32, %b: i32)
    -> !ondsp.acc<storage = i40, frac = 30, unsigned, update_overflow = saturate> {
  // expected-error@+1 {{fixed product semantics currently require a signed numeric policy}}
  %0 = ondsp.mac %acc, %a, %b {numeric = #ondsp.fixed<unsigned, storage = i32, frac = 31>, product = #ondsp.product<high_raw>} : (!ondsp.acc<storage = i40, frac = 30, unsigned, update_overflow = saturate>, i32, i32) -> !ondsp.acc<storage = i40, frac = 30, unsigned, update_overflow = saturate>
  return %0 : !ondsp.acc<storage = i40, frac = 30, unsigned, update_overflow = saturate>
}
