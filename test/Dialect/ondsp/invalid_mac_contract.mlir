// RUN: ondrix-opt %s -split-input-file -verify-diagnostics

func.func @wrong_full_product_frac(
    %acc: !ondsp.acc<storage = i40, frac = 29, signed>, %a: i16, %b: i16)
    -> !ondsp.acc<storage = i40, frac = 29, signed> {
  // expected-error@+1 {{accumulator frac 29 does not match expected frac 30 for full product}}
  %0 = ondsp.mac %acc, %a, %b {numeric = #ondsp.fixed<signed, storage = i16, frac = 15>, product = #ondsp.product<full>} : (!ondsp.acc<storage = i40, frac = 29, signed>, i16, i16) -> !ondsp.acc<storage = i40, frac = 29, signed>
  return %0 : !ondsp.acc<storage = i40, frac = 29, signed>
}

// -----

func.func @fixed_reduce_without_product(%a: i16, %b: i16) -> i32 {
  // expected-error@+1 {{fixed numeric policy requires a product attribute}}
  %0 = ondsp.reduce_mac %a, %b {numeric = #ondsp.fixed<signed, storage = i16, frac = 15>} : (i16, i16) -> i32
  return %0 : i32
}

// -----

func.func @fp_reduce_with_integer_product(%a: f32, %b: f32) -> f32 {
  // expected-error@+1 {{floating-point numeric policy must not specify a product attribute}}
  %0 = ondsp.reduce_mac %a, %b {numeric = #ondsp.fp<format = f32, contract = fma>, product = #ondsp.product<full>} : (f32, f32) -> f32
  return %0 : f32
}

// -----

func.func @negative_high_product_frac(
    %acc: !ondsp.acc<storage = i40, frac = 0, signed>, %a: i16, %b: i16)
    -> !ondsp.acc<storage = i40, frac = 0, signed> {
  // expected-error@+1 {{high product fractional position would be negative}}
  %0 = ondsp.mac %acc, %a, %b {numeric = #ondsp.fixed<signed, storage = i16, frac = 7>, product = #ondsp.product<high>} : (!ondsp.acc<storage = i40, frac = 0, signed>, i16, i16) -> !ondsp.acc<storage = i40, frac = 0, signed>
  return %0 : !ondsp.acc<storage = i40, frac = 0, signed>
}

// -----

func.func @accumulator_signedness_mismatch(
    %acc: !ondsp.acc<storage = i40, frac = 30, unsigned>, %a: i16, %b: i16)
    -> !ondsp.acc<storage = i40, frac = 30, unsigned> {
  // expected-error@+1 {{accumulator signedness must match the fixed numeric policy}}
  %0 = ondsp.mac %acc, %a, %b {numeric = #ondsp.fixed<signed, storage = i16, frac = 15>, product = #ondsp.product<full>} : (!ondsp.acc<storage = i40, frac = 30, unsigned>, i16, i16) -> !ondsp.acc<storage = i40, frac = 30, unsigned>
  return %0 : !ondsp.acc<storage = i40, frac = 30, unsigned>
}

// -----

func.func @fixed_reduce_noninteger_result(%a: i16, %b: i16) -> f32 {
  // expected-error@+1 {{fixed reduce_mac result must be a signless integer type of at least 32 bits}}
  %0 = ondsp.reduce_mac %a, %b {numeric = #ondsp.fixed<signed, storage = i16, frac = 15>, product = #ondsp.product<full>} : (i16, i16) -> f32
  return %0 : f32
}

// -----

func.func @fixed_reduce_narrow_result(%a: i16, %b: i16) -> i16 {
  // expected-error@+1 {{fixed reduce_mac result must be a signless integer type of at least 32 bits}}
  %0 = ondsp.reduce_mac %a, %b {numeric = #ondsp.fixed<signed, storage = i16, frac = 15>, product = #ondsp.product<full>} : (i16, i16) -> i16
  return %0 : i16
}

// -----

func.func @acc_extract_scale_result_mismatch(
    %acc: !ondsp.acc<storage = i40, frac = 30, signed>) -> i16 {
  // expected-error@+1 {{scale saturate_to type must match the result type}}
  %0 = ondsp.acc_extract %acc {scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 15, rounding = toward_negative, overflow = saturate, saturate_to = i32>} : (!ondsp.acc<storage = i40, frac = 30, signed>) -> i16
  return %0 : i16
}

// -----

func.func @reduce_mac_rejects_mixed_domains(
    %lhs: memref<8xf32>, %rhs: f32) -> f32 {
  // expected-error@+1 {{requires either two scalar operands or two rank-1 shaped operands}}
  %0 = ondsp.reduce_mac %lhs, %rhs {numeric = #ondsp.fp<format = f32, contract = fma>} : (memref<8xf32>, f32) -> f32
  return %0 : f32
}

// -----

func.func @reduce_mac_rejects_unranked(
    %lhs: memref<*xf32>, %rhs: memref<*xf32>) -> f32 {
  // expected-error@+1 {{shaped operands must be rank-1}}
  %0 = ondsp.reduce_mac %lhs, %rhs {numeric = #ondsp.fp<format = f32, contract = fma>} : (memref<*xf32>, memref<*xf32>) -> f32
  return %0 : f32
}

// -----

func.func @floating_reduce_mac_requires_scalar_result(
    %lhs: vector<8xf32>, %rhs: vector<8xf32>) -> vector<8xf32> {
  // expected-error@+1 {{floating-point reduce_mac result must be scalar and match numeric format}}
  %0 = ondsp.reduce_mac %lhs, %rhs {numeric = #ondsp.fp<format = f32, contract = fma>} : (vector<8xf32>, vector<8xf32>) -> vector<8xf32>
  return %0 : vector<8xf32>
}

// -----

func.func @reduce_mac_rejects_scalable_vectors(
    %lhs: vector<8xf32>, %rhs: vector<[8]xf32>) -> f32 {
  // expected-error@+1 {{scalable vector operands are not supported}}
  %0 = ondsp.reduce_mac %lhs, %rhs {numeric = #ondsp.fp<format = f32, contract = fma>} : (vector<8xf32>, vector<[8]xf32>) -> f32
  return %0 : f32
}

// -----

func.func @fixed_reduce_requires_signless_result(%a: i16, %b: i16) -> si32 {
  // expected-error@+1 {{fixed reduce_mac result must be a signless integer type of at least 32 bits}}
  %0 = ondsp.reduce_mac %a, %b {numeric = #ondsp.fixed<signed, storage = i16, frac = 15>, product = #ondsp.product<full>} : (i16, i16) -> si32
  return %0 : si32
}
