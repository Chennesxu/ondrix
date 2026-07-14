// RUN: ondrix-opt %s -split-input-file -verify-diagnostics

func.func @assume_numeric_rejects_memref(
    %input: memref<1xi16>) -> memref<1xi16> {
  // expected-error@+1 {{value-only operation does not accept memref operands}}
  %0 = ondsp.assume_numeric %input {numeric = #ondsp.fixed<signed, storage = i16, frac = 15>} : (memref<1xi16>) -> memref<1xi16>
  return %0 : memref<1xi16>
}

// -----

func.func @convert_rejects_unranked_memref(
    %input: memref<*xi16>) -> i8 {
  // expected-error@+1 {{value-only operation does not accept memref operands}}
  %0 = ondsp.convert %input {src = #ondsp.fixed<signed, storage = i16, frac = 15>, dst = #ondsp.fixed<signed, storage = i8, frac = 7>} : (memref<*xi16>) -> i8
  return %0 : i8
}

// -----

func.func @round_shift_rejects_memref_result(
    %input: i16) -> memref<1xi16> {
  // expected-error@+1 {{value-only operation does not produce memref results}}
  %0 = ondsp.round_shift %input {scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 1, rounding = trunc, overflow = wrap, saturate_to = i16>} : (i16) -> memref<1xi16>
  return %0 : memref<1xi16>
}

// -----

func.func @sat_cast_rejects_unranked_memref_result(
    %input: i32) -> memref<*xi16> {
  // expected-error@+1 {{value-only operation does not produce memref results}}
  %0 = ondsp.sat_cast %input {numeric = #ondsp.fixed<signed, storage = i16, frac = 15>} : (i32) -> memref<*xi16>
  return %0 : memref<*xi16>
}

// -----

func.func @sat_add_shift_rejects_memref_operand(
    %lhs: memref<1xi16>, %rhs: i16) -> i16 {
  // expected-error@+1 {{value-only operation does not accept memref operands}}
  %0 = ondsp.sat_add_shift %lhs, %rhs {scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 1, rounding = trunc, overflow = saturate, saturate_to = i16>} : (memref<1xi16>, i16) -> i16
  return %0 : i16
}

// -----

func.func @sat_sub_shift_rejects_memref_result(
    %lhs: i16, %rhs: i16) -> memref<1xi16> {
  // expected-error@+1 {{value-only operation does not produce memref results}}
  %0 = ondsp.sat_sub_shift %lhs, %rhs {scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 1, rounding = trunc, overflow = saturate, saturate_to = i16>} : (i16, i16) -> memref<1xi16>
  return %0 : memref<1xi16>
}

// -----

func.func @acc_init_rejects_unranked_memref(
    %input: memref<*xi16>) -> !ondsp.acc<storage = i40, frac = 30, signed> {
  // expected-error@+1 {{value-only operation does not accept memref operands}}
  %0 = ondsp.acc_init %input : (memref<*xi16>) -> !ondsp.acc<storage = i40, frac = 30, signed>
  return %0 : !ondsp.acc<storage = i40, frac = 30, signed>
}

// -----

func.func @numeric_value_ops_accept_shaped_values(
    %tensor: tensor<2xi16>, %vector: vector<2xi16>)
    -> (tensor<2xi16>, vector<2xi8>) {
  %0 = ondsp.assume_numeric %tensor {numeric = #ondsp.fixed<signed, storage = i16, frac = 15>} : (tensor<2xi16>) -> tensor<2xi16>
  %1 = ondsp.convert %vector {src = #ondsp.fixed<signed, storage = i16, frac = 15>, dst = #ondsp.fixed<signed, storage = i8, frac = 7>} : (vector<2xi16>) -> vector<2xi8>
  return %0, %1 : tensor<2xi16>, vector<2xi8>
}

// -----

func.func @elementwise_value_ops_accept_shaped_values(
    %tensor: tensor<2xi16>, %vector: vector<2xi16>)
    -> (vector<2xi16>, tensor<2xi16>, vector<2xi16>, tensor<2xi16>) {
  %0 = ondsp.round_shift %vector {scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 1, rounding = trunc, overflow = wrap, saturate_to = i16>} : (vector<2xi16>) -> vector<2xi16>
  %1 = ondsp.sat_cast %tensor {numeric = #ondsp.fixed<signed, storage = i16, frac = 15>} : (tensor<2xi16>) -> tensor<2xi16>
  %2 = ondsp.sat_add_shift %vector, %vector {scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 1, rounding = trunc, overflow = saturate, saturate_to = i16>} : (vector<2xi16>, vector<2xi16>) -> vector<2xi16>
  %3 = ondsp.sat_sub_shift %tensor, %tensor {scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 1, rounding = trunc, overflow = saturate, saturate_to = i16>} : (tensor<2xi16>, tensor<2xi16>) -> tensor<2xi16>
  return %0, %1, %2, %3 : vector<2xi16>, tensor<2xi16>, vector<2xi16>, tensor<2xi16>
}
