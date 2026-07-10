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
