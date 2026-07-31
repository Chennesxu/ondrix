// RUN: ondrix-opt %s -split-input-file -verify-diagnostics

func.func @quantize_rejects_invalid_source_numeric(%input: i32) -> i16 {
  // expected-error@+1 {{attribute 'src' failed to satisfy constraint: ondsp fixed-point or floating-point numeric attribute}}
  %0 = ondrix.quantize %input {src = "invalid", dst = #ondsp.fixed<signed, storage = i16, frac = 15>} : (i32) -> i16
  return %0 : i16
}

// -----

func.func @quantize_rejects_invalid_destination_numeric(%input: i32) -> i16 {
  // expected-error@+1 {{attribute 'dst' failed to satisfy constraint: ondsp fixed-point or floating-point numeric attribute}}
  %0 = ondrix.quantize %input {src = #ondsp.fixed<signed, storage = i32, frac = 30>, dst = "invalid"} : (i32) -> i16
  return %0 : i16
}

// -----

func.func @butterfly_rejects_invalid_layout(
    %a: i32, %b: i32, %twiddle: i32) -> (i32, i32) {
  // expected-error@+1 {{attribute 'layout' failed to satisfy constraint: Complex value storage layout}}
  %0, %1 = ondrix.butterfly %a, %b, %twiddle {layout = "invalid", numeric = #ondsp.fixed<signed, storage = i16, frac = 15>, product = #ondsp.product<full>, product_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 15, rounding = nearest_even, overflow = saturate, saturate_to = i16>, output_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 1, rounding = nearest_even, overflow = saturate, saturate_to = i16>} : (i32, i32, i32) -> (i32, i32)
  return %0, %1 : i32, i32
}

// -----

func.func @butterfly_rejects_invalid_numeric(
    %a: i32, %b: i32, %twiddle: i32) -> (i32, i32) {
  // expected-error@+1 {{attribute 'numeric' failed to satisfy constraint: ondsp fixed-point or floating-point numeric attribute}}
  %0, %1 = ondrix.butterfly %a, %b, %twiddle {layout = #ondsp.cx_layout<packed_i16_imag_hi_real_lo>, numeric = "invalid", product = #ondsp.product<full>, product_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 15, rounding = nearest_even, overflow = saturate, saturate_to = i16>, output_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 1, rounding = nearest_even, overflow = saturate, saturate_to = i16>} : (i32, i32, i32) -> (i32, i32)
  return %0, %1 : i32, i32
}

// -----

// The value domain runs first, so this case carries an otherwise valid packed
// layout and scalar i32 values; the unsigned numeric policy is what must be
// reported. The layout and value-shape rejections are separate cases below.
func.func @butterfly_rejects_unsigned_product_contract(%a: i32, %b: i32, %twiddle: i32)
    -> (i32, i32) {
  // expected-error@+1 {{packed butterfly requires signed Q15 numeric semantics for this layout}}
  %0, %1 = ondrix.butterfly %a, %b, %twiddle {layout = #ondsp.cx_layout<packed_i16_imag_hi_real_lo>, numeric = #ondsp.fixed<unsigned, storage = i16, frac = 15>, product = #ondsp.product<full>, product_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 15, rounding = nearest_even, overflow = saturate, saturate_to = i16>, output_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 1, rounding = nearest_even, overflow = saturate, saturate_to = i16>} : (i32, i32, i32) -> (i32, i32)
  return %0, %1 : i32, i32
}

// -----

func.func @butterfly_rejects_interleaved_layout(%a: i32, %b: i32, %twiddle: i32) -> (i32, i32) {
  // expected-error@+1 {{executable butterfly requires packed_i16_imag_hi_real_lo layout}}
  %0, %1 = ondrix.butterfly %a, %b, %twiddle {layout = #ondsp.cx_layout<interleaved>, numeric = #ondsp.fixed<signed, storage = i16, frac = 15>, product = #ondsp.product<full>, product_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 15, rounding = nearest_even, overflow = saturate, saturate_to = i16>, output_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 1, rounding = nearest_even, overflow = saturate, saturate_to = i16>} : (i32, i32, i32) -> (i32, i32)
  return %0, %1 : i32, i32
}

// -----

// The algorithm-level butterfly stays scalar; lane batching is a lowering
// decision, so a Vector value domain is rejected here even under the packed
// layout the executable profile uses.
func.func @butterfly_rejects_vector_values(
    %a: vector<2xi16>, %b: vector<2xi16>, %twiddle: vector<2xi16>)
    -> (vector<2xi16>, vector<2xi16>) {
  // expected-error@+1 {{executable butterfly requires scalar signless i32 packed values}}
  %0, %1 = ondrix.butterfly %a, %b, %twiddle {layout = #ondsp.cx_layout<packed_i16_imag_hi_real_lo>, numeric = #ondsp.fixed<signed, storage = i16, frac = 15>, product = #ondsp.product<full>, product_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 15, rounding = nearest_even, overflow = saturate, saturate_to = i16>, output_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 1, rounding = nearest_even, overflow = saturate, saturate_to = i16>} : (vector<2xi16>, vector<2xi16>, vector<2xi16>) -> (vector<2xi16>, vector<2xi16>)
  return %0, %1 : vector<2xi16>, vector<2xi16>
}
