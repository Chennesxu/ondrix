// RUN: ondrix-opt %s -split-input-file -verify-diagnostics

func.func @packed_butterfly_non_i32_container(
    %a: vector<4xi16>, %b: vector<4xi16>, %tw: vector<4xi16>)
    -> (vector<4xi16>, vector<4xi16>) {
  // expected-error@+1 {{packed i16 butterfly operands must use i32 container storage}}
  %0, %1 = ondsp.cx_butterfly %a, %b, %tw {layout = #ondsp.cx_layout<packed_i16_imag_hi_real_lo>, numeric = #ondsp.fixed<signed, storage = i16, frac = 15>, product = #ondsp.product<full>, scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 15, rounding = trunc, overflow = saturate, saturate_to = i16>} : (vector<4xi16>, vector<4xi16>, vector<4xi16>) -> (vector<4xi16>, vector<4xi16>)
  return %0, %1 : vector<4xi16>, vector<4xi16>
}

// -----

func.func @fixed_butterfly_missing_scale(%a: i32, %b: i32, %tw: i32) -> (i32, i32) {
  // expected-error@+1 {{fixed numeric butterfly requires a scale attribute}}
  %0, %1 = ondsp.cx_butterfly %a, %b, %tw {layout = #ondsp.cx_layout<packed_i16_imag_hi_real_lo>, numeric = #ondsp.fixed<signed, storage = i16, frac = 15>, product = #ondsp.product<full>} : (i32, i32, i32) -> (i32, i32)
  return %0, %1 : i32, i32
}

// -----

func.func @floating_butterfly_with_scale(%a: f32, %b: f32, %tw: f32) -> (f32, f32) {
  // expected-error@+1 {{floating-point numeric butterfly must not specify a scale attribute}}
  %0, %1 = ondsp.cx_butterfly %a, %b, %tw {layout = #ondsp.cx_layout<split>, numeric = #ondsp.fp<format = f32, contract = off>, scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 0, rounding = trunc, overflow = wrap, saturate_to = i32>} : (f32, f32, f32) -> (f32, f32)
  return %0, %1 : f32, f32
}

// -----

func.func @split_butterfly_fp_storage_mismatch(%a: f64, %b: f64, %tw: f64)
    -> (f64, f64) {
  // expected-error@+1 {{a type does not match numeric storage type}}
  %0, %1 = ondsp.cx_butterfly %a, %b, %tw {layout = #ondsp.cx_layout<split>, numeric = #ondsp.fp<format = f32, contract = off>} : (f64, f64, f64) -> (f64, f64)
  return %0, %1 : f64, f64
}

// -----

func.func @interleaved_butterfly_fixed_storage_mismatch(
    %a: vector<2xi32>, %b: vector<2xi32>, %tw: vector<2xi32>)
    -> (vector<2xi32>, vector<2xi32>) {
  // expected-error@+1 {{a type does not match numeric storage type}}
  %0, %1 = ondsp.cx_butterfly %a, %b, %tw {layout = #ondsp.cx_layout<interleaved>, numeric = #ondsp.fixed<signed, storage = i16, frac = 15>, product = #ondsp.product<full>, scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 15, rounding = trunc, overflow = saturate, saturate_to = i16>} : (vector<2xi32>, vector<2xi32>, vector<2xi32>) -> (vector<2xi32>, vector<2xi32>)
  return %0, %1 : vector<2xi32>, vector<2xi32>
}
