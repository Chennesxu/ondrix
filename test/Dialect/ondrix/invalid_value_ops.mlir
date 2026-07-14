// RUN: ondrix-opt %s -split-input-file -verify-diagnostics

func.func @butterfly_rejects_memref_operand(
    %a: memref<1xi32>, %b: i32, %tw: i32) -> (i32, i32) {
  // expected-error@+1 {{value-only operation does not accept memref operands}}
  %0, %1 = ondrix.butterfly %a, %b, %tw {layout = #ondsp.cx_layout<packed_i16_imag_hi_real_lo>, numeric = #ondsp.fixed<signed, storage = i16, frac = 15>, product = #ondsp.product<full>, scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 15, rounding = toward_negative, overflow = saturate, saturate_to = i16>} : (memref<1xi32>, i32, i32) -> (i32, i32)
  return %0, %1 : i32, i32
}

// -----

func.func @butterfly_rejects_unranked_memref_operand(
    %a: memref<*xi32>, %b: i32, %tw: i32) -> (i32, i32) {
  // expected-error@+1 {{value-only operation does not accept memref operands}}
  %0, %1 = ondrix.butterfly %a, %b, %tw {layout = #ondsp.cx_layout<packed_i16_imag_hi_real_lo>, numeric = #ondsp.fixed<signed, storage = i16, frac = 15>, product = #ondsp.product<full>, scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 15, rounding = toward_negative, overflow = saturate, saturate_to = i16>} : (memref<*xi32>, i32, i32) -> (i32, i32)
  return %0, %1 : i32, i32
}

// -----

func.func @butterfly_rejects_memref_result(
    %a: i32, %b: i32, %tw: i32) -> (memref<1xi32>, i32) {
  // expected-error@+1 {{value-only operation does not produce memref results}}
  %0, %1 = ondrix.butterfly %a, %b, %tw {layout = #ondsp.cx_layout<packed_i16_imag_hi_real_lo>, numeric = #ondsp.fixed<signed, storage = i16, frac = 15>, product = #ondsp.product<full>, scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 15, rounding = toward_negative, overflow = saturate, saturate_to = i16>} : (i32, i32, i32) -> (memref<1xi32>, i32)
  return %0, %1 : memref<1xi32>, i32
}

// -----

func.func @butterfly_rejects_unranked_memref_result(
    %a: i32, %b: i32, %tw: i32) -> (memref<*xi32>, i32) {
  // expected-error@+1 {{value-only operation does not produce memref results}}
  %0, %1 = ondrix.butterfly %a, %b, %tw {layout = #ondsp.cx_layout<packed_i16_imag_hi_real_lo>, numeric = #ondsp.fixed<signed, storage = i16, frac = 15>, product = #ondsp.product<full>, scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 15, rounding = toward_negative, overflow = saturate, saturate_to = i16>} : (i32, i32, i32) -> (memref<*xi32>, i32)
  return %0, %1 : memref<*xi32>, i32
}

// -----

func.func @quantize_rejects_memref_operand(
    %input: memref<1xi32>) -> i16 {
  // expected-error@+1 {{value-only operation does not accept memref operands}}
  %0 = ondrix.quantize %input {src = #ondsp.fixed<signed, storage = i32, frac = 30>, dst = #ondsp.fixed<signed, storage = i16, frac = 15>} : (memref<1xi32>) -> i16
  return %0 : i16
}

// -----

func.func @quantize_rejects_unranked_memref_operand(
    %input: memref<*xi32>) -> i16 {
  // expected-error@+1 {{value-only operation does not accept memref operands}}
  %0 = ondrix.quantize %input {src = #ondsp.fixed<signed, storage = i32, frac = 30>, dst = #ondsp.fixed<signed, storage = i16, frac = 15>} : (memref<*xi32>) -> i16
  return %0 : i16
}

// -----

func.func @quantize_rejects_memref_result(
    %input: i32) -> memref<1xi16> {
  // expected-error@+1 {{value-only operation does not produce memref results}}
  %0 = ondrix.quantize %input {src = #ondsp.fixed<signed, storage = i32, frac = 30>, dst = #ondsp.fixed<signed, storage = i16, frac = 15>} : (i32) -> memref<1xi16>
  return %0 : memref<1xi16>
}

// -----

func.func @quantize_rejects_unranked_memref_result(
    %input: i32) -> memref<*xi16> {
  // expected-error@+1 {{value-only operation does not produce memref results}}
  %0 = ondrix.quantize %input {src = #ondsp.fixed<signed, storage = i32, frac = 30>, dst = #ondsp.fixed<signed, storage = i16, frac = 15>} : (i32) -> memref<*xi16>
  return %0 : memref<*xi16>
}

// -----

func.func @butterfly_rejects_nested_memref(
    %a: tuple<memref<1xi32>>, %b: i32, %tw: i32) -> (i32, i32) {
  // expected-error@+1 {{value-only operation does not accept memref operands}}
  %0, %1 = ondrix.butterfly %a, %b, %tw {layout = #ondsp.cx_layout<packed_i16_imag_hi_real_lo>, numeric = #ondsp.fixed<signed, storage = i16, frac = 15>, product = #ondsp.product<full>, scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 15, rounding = toward_negative, overflow = saturate, saturate_to = i16>} : (tuple<memref<1xi32>>, i32, i32) -> (i32, i32)
  return %0, %1 : i32, i32
}

// -----

func.func @butterfly_accepts_vector_values(
    %a: vector<2xi16>, %b: vector<2xi16>, %tw: vector<2xi16>)
    -> (vector<2xi16>, vector<2xi16>) {
  %0, %1 = ondrix.butterfly %a, %b, %tw {layout = #ondsp.cx_layout<interleaved>, numeric = #ondsp.fixed<signed, storage = i16, frac = 15>, product = #ondsp.product<full>, scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 15, rounding = toward_negative, overflow = saturate, saturate_to = i16>} : (vector<2xi16>, vector<2xi16>, vector<2xi16>) -> (vector<2xi16>, vector<2xi16>)
  return %0, %1 : vector<2xi16>, vector<2xi16>
}

// -----

func.func @quantize_accepts_tensor_values(
    %input: tensor<2xi32>) -> tensor<2xi16> {
  %0 = ondrix.quantize %input {src = #ondsp.fixed<signed, storage = i32, frac = 30>, dst = #ondsp.fixed<signed, storage = i16, frac = 15>} : (tensor<2xi32>) -> tensor<2xi16>
  return %0 : tensor<2xi16>
}
