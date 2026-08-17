// RUN: ondrix-opt %s -split-input-file -verify-diagnostics

func.func @rejects_non_q15_numeric(%a: i32, %b: i32, %tw: i32) -> (i32, i32) {
  // expected-error@+1 {{packed butterfly requires signed Q15 numeric semantics for this layout}}
  %0, %1 = ondsp.cx_butterfly %a, %b, %tw {
    layout = #ondsp.cx_layout<packed_i16_imag_hi_real_lo>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 14>,
    product = #ondsp.product<full>,
    product_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 15, rounding = nearest_even, overflow = saturate, saturate_to = i16>,
    output_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 1, rounding = nearest_even, overflow = saturate, saturate_to = i16>
  } : (i32, i32, i32) -> (i32, i32)
  return %0, %1 : i32, i32
}

// -----

func.func @rejects_raw_high_product(%a: i32, %b: i32, %tw: i32) -> (i32, i32) {
  // expected-error@+1 {{packed butterfly requires product = #ondsp.product<full>}}
  %0, %1 = ondsp.cx_butterfly %a, %b, %tw {
    layout = #ondsp.cx_layout<packed_i16_imag_hi_real_lo>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    product = #ondsp.product<high_raw>,
    product_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 15, rounding = nearest_even, overflow = saturate, saturate_to = i16>,
    output_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 1, rounding = nearest_even, overflow = saturate, saturate_to = i16>
  } : (i32, i32, i32) -> (i32, i32)
  return %0, %1 : i32, i32
}

// -----

func.func @rejects_other_layout(%a: i32, %b: i32, %tw: i32) -> (i32, i32) {
  // expected-error@+1 {{executable butterfly requires packed_i16_imag_hi_real_lo or packed_i32_imag_hi_real_lo layout}}
  %0, %1 = ondsp.cx_butterfly %a, %b, %tw {
    layout = #ondsp.cx_layout<packed_i16_real_hi_imag_lo>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    product = #ondsp.product<full>,
    product_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 15, rounding = nearest_even, overflow = saturate, saturate_to = i16>,
    output_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 1, rounding = nearest_even, overflow = saturate, saturate_to = i16>
  } : (i32, i32, i32) -> (i32, i32)
  return %0, %1 : i32, i32
}

// -----

func.func @rejects_mismatched_vector_shapes(
    %a: vector<2xi32>, %b: vector<4xi32>, %tw: vector<2xi32>)
    -> (vector<2xi32>, vector<2xi32>) {
  // expected-error@+1 {{operands and results must use the same scalar or static shaped domain}}
  %0, %1 = ondsp.cx_butterfly %a, %b, %tw {
    layout = #ondsp.cx_layout<packed_i16_imag_hi_real_lo>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    product = #ondsp.product<full>,
    product_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 15, rounding = nearest_even, overflow = saturate, saturate_to = i16>,
    output_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 1, rounding = nearest_even, overflow = saturate, saturate_to = i16>
  } : (vector<2xi32>, vector<4xi32>, vector<2xi32>)
      -> (vector<2xi32>, vector<2xi32>)
  return %0, %1 : vector<2xi32>, vector<2xi32>
}

// -----

func.func @rejects_product_shift(%a: i32, %b: i32, %tw: i32) -> (i32, i32) {
  // expected-error@+1 {{product_scale requires pre_shift_left=0 and post_shift_right=15}}
  %0, %1 = ondsp.cx_butterfly %a, %b, %tw {
    layout = #ondsp.cx_layout<packed_i16_imag_hi_real_lo>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    product = #ondsp.product<full>,
    product_scale = #ondsp.scale<pre_shift_left = 1, post_shift_right = 15, rounding = nearest_even, overflow = saturate, saturate_to = i16>,
    output_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 1, rounding = nearest_even, overflow = saturate, saturate_to = i16>
  } : (i32, i32, i32) -> (i32, i32)
  return %0, %1 : i32, i32
}

// -----

func.func @rejects_product_rounding(%a: i32, %b: i32, %tw: i32) -> (i32, i32) {
  // expected-error@+1 {{product_scale admits nearest_even, toward_negative, or nearest_ties_positive rounding}}
  %0, %1 = ondsp.cx_butterfly %a, %b, %tw {
    layout = #ondsp.cx_layout<packed_i16_imag_hi_real_lo>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    product = #ondsp.product<full>,
    product_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 15, rounding = toward_zero, overflow = saturate, saturate_to = i16>,
    output_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 1, rounding = nearest_even, overflow = saturate, saturate_to = i16>
  } : (i32, i32, i32) -> (i32, i32)
  return %0, %1 : i32, i32
}

// -----

func.func @rejects_output_shift(%a: i32, %b: i32, %tw: i32) -> (i32, i32) {
  // expected-error@+1 {{output_scale requires pre_shift_left=0 and post_shift_right=0 or 1}}
  %0, %1 = ondsp.cx_butterfly %a, %b, %tw {
    layout = #ondsp.cx_layout<packed_i16_imag_hi_real_lo>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    product = #ondsp.product<full>,
    product_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 15, rounding = nearest_even, overflow = saturate, saturate_to = i16>,
    output_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 2, rounding = nearest_even, overflow = saturate, saturate_to = i16>
  } : (i32, i32, i32) -> (i32, i32)
  return %0, %1 : i32, i32
}

// -----

func.func @rejects_output_rounding(%a: i32, %b: i32, %tw: i32) -> (i32, i32) {
  // expected-error@+1 {{output_scale admits nearest_even, toward_negative, or nearest_ties_positive rounding}}
  %0, %1 = ondsp.cx_butterfly %a, %b, %tw {
    layout = #ondsp.cx_layout<packed_i16_imag_hi_real_lo>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    product = #ondsp.product<full>,
    product_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 15, rounding = nearest_even, overflow = saturate, saturate_to = i16>,
    output_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 1, rounding = toward_zero, overflow = saturate, saturate_to = i16>
  } : (i32, i32, i32) -> (i32, i32)
  return %0, %1 : i32, i32
}

// -----

func.func @q31_rejects_i32_container(%a: i32, %b: i32, %tw: i32) -> (i32, i32) {
  // expected-error@+1 {{executable butterfly requires scalar or fixed Vector signless i64 packed values}}
  %0, %1 = ondsp.cx_butterfly %a, %b, %tw {
    layout = #ondsp.cx_layout<packed_i32_imag_hi_real_lo>,
    numeric = #ondsp.fixed<signed, storage = i32, frac = 31>,
    product = #ondsp.product<full>,
    product_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 31, rounding = nearest_even, overflow = saturate, saturate_to = i32>,
    output_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 1, rounding = nearest_even, overflow = saturate, saturate_to = i32>
  } : (i32, i32, i32) -> (i32, i32)
  return %0, %1 : i32, i32
}

// -----

func.func @q15_layout_rejects_i64_container(%a: i64, %b: i64, %tw: i64) -> (i64, i64) {
  // expected-error@+1 {{executable butterfly requires scalar or fixed Vector signless i32 packed values}}
  %0, %1 = ondsp.cx_butterfly %a, %b, %tw {
    layout = #ondsp.cx_layout<packed_i16_imag_hi_real_lo>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    product = #ondsp.product<full>,
    product_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 15, rounding = nearest_even, overflow = saturate, saturate_to = i16>,
    output_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 1, rounding = nearest_even, overflow = saturate, saturate_to = i16>
  } : (i64, i64, i64) -> (i64, i64)
  return %0, %1 : i64, i64
}

// -----

func.func @q31_rejects_q15_numeric(%a: i64, %b: i64, %tw: i64) -> (i64, i64) {
  // expected-error@+1 {{packed butterfly requires signed Q31 numeric semantics for this layout}}
  %0, %1 = ondsp.cx_butterfly %a, %b, %tw {
    layout = #ondsp.cx_layout<packed_i32_imag_hi_real_lo>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    product = #ondsp.product<full>,
    product_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 31, rounding = nearest_even, overflow = saturate, saturate_to = i32>,
    output_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 1, rounding = nearest_even, overflow = saturate, saturate_to = i32>
  } : (i64, i64, i64) -> (i64, i64)
  return %0, %1 : i64, i64
}

// -----

func.func @q31_rejects_target_inventory(%a: i64, %b: i64, %tw: i64) -> (i64, i64) {
  // expected-error@+1 {{product_scale requires nearest_even rounding}}
  %0, %1 = ondsp.cx_butterfly %a, %b, %tw {
    layout = #ondsp.cx_layout<packed_i32_imag_hi_real_lo>,
    numeric = #ondsp.fixed<signed, storage = i32, frac = 31>,
    product = #ondsp.product<full>,
    product_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 31, rounding = toward_negative, overflow = wrap, saturate_to = i32>,
    output_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 1, rounding = nearest_even, overflow = saturate, saturate_to = i32>
  } : (i64, i64, i64) -> (i64, i64)
  return %0, %1 : i64, i64
}

// -----

func.func @cross_rejects_nearest_even(%a: i32, %b: i32, %tw: i32) -> (i32, i32) {
  // expected-error@+1 {{the cross and unit combines are admitted only under target-inventory toward_negative or nearest_ties_positive rounding}}
  %0, %1 = ondsp.cx_butterfly %a, %b, %tw {
    layout = #ondsp.cx_layout<packed_i16_imag_hi_real_lo>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    product = #ondsp.product<full>,
    product_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 15, rounding = nearest_even, overflow = saturate, saturate_to = i16>,
    output_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 1, rounding = nearest_even, overflow = saturate, saturate_to = i16>,
    variant = #ondsp.cx_butterfly_variant<cross>
  } : (i32, i32, i32) -> (i32, i32)
  return %0, %1 : i32, i32
}

// -----

func.func @cross_rejects_vector_shape(%a: vector<4xi32>, %b: vector<4xi32>, %tw: vector<4xi32>) -> (vector<4xi32>, vector<4xi32>) {
  // expected-error@+1 {{the cross and unit combines are admitted only on scalar packed Q15 values}}
  %0, %1 = ondsp.cx_butterfly %a, %b, %tw {
    layout = #ondsp.cx_layout<packed_i16_imag_hi_real_lo>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    product = #ondsp.product<full>,
    product_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 15, rounding = toward_negative, overflow = wrap, saturate_to = i16>,
    output_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 1, rounding = toward_negative, overflow = wrap, saturate_to = i16>,
    variant = #ondsp.cx_butterfly_variant<cross>
  } : (vector<4xi32>, vector<4xi32>, vector<4xi32>) -> (vector<4xi32>, vector<4xi32>)
  return %0, %1 : vector<4xi32>, vector<4xi32>
}

// -----

func.func @cross_rejects_q31(%a: i64, %b: i64, %tw: i64) -> (i64, i64) {
  // expected-error@+1 {{the cross and unit combines are admitted only on scalar packed Q15 values}}
  %0, %1 = ondsp.cx_butterfly %a, %b, %tw {
    layout = #ondsp.cx_layout<packed_i32_imag_hi_real_lo>,
    numeric = #ondsp.fixed<signed, storage = i32, frac = 31>,
    product = #ondsp.product<full>,
    product_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 31, rounding = nearest_even, overflow = saturate, saturate_to = i32>,
    output_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 1, rounding = nearest_even, overflow = saturate, saturate_to = i32>,
    variant = #ondsp.cx_butterfly_variant<cross>
  } : (i64, i64, i64) -> (i64, i64)
  return %0, %1 : i64, i64
}

// -----

func.func @unit_rejects_nearest_even(%a: i32, %b: i32, %tw: i32) -> (i32, i32) {
  // expected-error@+1 {{the cross and unit combines are admitted only under target-inventory toward_negative or nearest_ties_positive rounding}}
  %0, %1 = ondsp.cx_butterfly %a, %b, %tw {
    layout = #ondsp.cx_layout<packed_i16_imag_hi_real_lo>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    product = #ondsp.product<full>,
    product_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 15, rounding = nearest_even, overflow = saturate, saturate_to = i16>,
    output_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 1, rounding = nearest_even, overflow = saturate, saturate_to = i16>,
    variant = #ondsp.cx_butterfly_variant<unit>
  } : (i32, i32, i32) -> (i32, i32)
  return %0, %1 : i32, i32
}

// -----

func.func @unit_cross_rejects_vector_shape(%a: vector<4xi32>, %b: vector<4xi32>, %tw: vector<4xi32>) -> (vector<4xi32>, vector<4xi32>) {
  // expected-error@+1 {{the cross and unit combines are admitted only on scalar packed Q15 values}}
  %0, %1 = ondsp.cx_butterfly %a, %b, %tw {
    layout = #ondsp.cx_layout<packed_i16_imag_hi_real_lo>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    product = #ondsp.product<full>,
    product_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 15, rounding = toward_negative, overflow = wrap, saturate_to = i16>,
    output_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 1, rounding = toward_negative, overflow = wrap, saturate_to = i16>,
    variant = #ondsp.cx_butterfly_variant<unit_cross>
  } : (vector<4xi32>, vector<4xi32>, vector<4xi32>) -> (vector<4xi32>, vector<4xi32>)
  return %0, %1 : vector<4xi32>, vector<4xi32>
}
