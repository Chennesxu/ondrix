// RUN: ondrix-opt %s -split-input-file -verify-diagnostics

func.func @dot_rejects_mixed_domains(%lhs: memref<8xf32>, %rhs: f32) -> f32 {
  // expected-error@+1 {{requires either two scalar operands or two rank-1 shaped operands}}
  %0 = ondrix.dot %lhs, %rhs {numeric = #ondsp.fp<format = f32, contract = off>} : (memref<8xf32>, f32) -> f32
  return %0 : f32
}

// -----

func.func @dot_rejects_tensor_operands(
    %lhs: tensor<8xf32>, %rhs: tensor<8xf32>) -> f32 {
  // expected-error@+1 {{tensor dot operands have no executable consumer; use memrefs or fixed vectors}}
  %0 = ondrix.dot %lhs, %rhs {numeric = #ondsp.fp<format = f32, contract = off>} : (tensor<8xf32>, tensor<8xf32>) -> f32
  return %0 : f32
}

// -----

func.func @dot_rejects_unranked(
    %lhs: memref<*xf32>, %rhs: memref<*xf32>) -> f32 {
  // expected-error@+1 {{shaped operands must be rank-1}}
  %0 = ondrix.dot %lhs, %rhs {numeric = #ondsp.fp<format = f32, contract = off>} : (memref<*xf32>, memref<*xf32>) -> f32
  return %0 : f32
}

// -----

func.func @dot_rejects_static_length_mismatch(
    %lhs: memref<8xf32>, %rhs: memref<4xf32>) -> f32 {
  // expected-error@+1 {{shaped operands must have equal static lengths}}
  %0 = ondrix.dot %lhs, %rhs {numeric = #ondsp.fp<format = f32, contract = off>} : (memref<8xf32>, memref<4xf32>) -> f32
  return %0 : f32
}

// -----

func.func @dot_requires_matching_element_types(
    %lhs: memref<8xf32>, %rhs: memref<8xf64>) -> f32 {
  // expected-error@+1 {{operand element types must match}}
  %0 = ondrix.dot %lhs, %rhs {numeric = #ondsp.fp<format = f32, contract = off>} : (memref<8xf32>, memref<8xf64>) -> f32
  return %0 : f32
}

// -----

func.func @fixed_dot_requires_numeric_storage(%lhs: i32, %rhs: i32) -> i32 {
  // expected-error@+1 {{operand element type must match fixed numeric storage type}}
  %0 = ondrix.dot %lhs, %rhs {numeric = #ondsp.fixed<signed, storage = i16, frac = 15>, product = #ondsp.product<full>} : (i32, i32) -> i32
  return %0 : i32
}

// -----

func.func @fixed_dot_requires_accumulator_result(%lhs: i16, %rhs: i16) -> i16 {
  // expected-error@+1 {{fixed reduction result must use !ondsp.acc}}
  %0 = ondrix.dot %lhs, %rhs {numeric = #ondsp.fixed<signed, storage = i16, frac = 15>, product = #ondsp.product<full>} : (i16, i16) -> i16
  return %0 : i16
}

// -----

func.func @floating_dot_requires_numeric_format(%lhs: f32, %rhs: f32) -> f64 {
  // expected-error@+1 {{floating-point dot operands and result must match numeric format}}
  %0 = ondrix.dot %lhs, %rhs {numeric = #ondsp.fp<format = f32, contract = off>} : (f32, f32) -> f64
  return %0 : f64
}

// -----

func.func @dot_rejects_scalable_vectors(
    %lhs: vector<8xf32>, %rhs: vector<[8]xf32>) -> f32 {
  // expected-error@+1 {{scalable vector operands are not supported}}
  %0 = ondrix.dot %lhs, %rhs {numeric = #ondsp.fp<format = f32, contract = off>} : (vector<8xf32>, vector<[8]xf32>) -> f32
  return %0 : f32
}

// -----

func.func @dot_rejects_floating_vectors(
    %lhs: vector<8xf32>, %rhs: vector<8xf32>) -> f32 {
  // expected-error@+1 {{floating-point vector dot operands have no executable consumer}}
  %0 = ondrix.dot %lhs, %rhs {numeric = #ondsp.fp<format = f32, contract = off>} : (vector<8xf32>, vector<8xf32>) -> f32
  return %0 : f32
}

// -----

func.func @fixed_dot_rejects_builtin_integer_result(%lhs: i16, %rhs: i16) -> ui32 {
  // expected-error@+1 {{fixed reduction result must use !ondsp.acc}}
  %0 = ondrix.dot %lhs, %rhs {numeric = #ondsp.fixed<signed, storage = i16, frac = 15>, product = #ondsp.product<full>} : (i16, i16) -> ui32
  return %0 : ui32
}

// -----

// The declared format is admitted per operation, not per attribute: FpAttr is
// format-parametric, but only f32 has a lowering and a differential reference.
func.func @dot_rejects_f64(%lhs: memref<8xf64>, %rhs: memref<8xf64>) -> f64 {
  // expected-error@+1 {{executable dot supports the f32 floating-point format}}
  %0 = ondrix.dot %lhs, %rhs {numeric = #ondsp.fp<format = f64, contract = off>} : (memref<8xf64>, memref<8xf64>) -> f64
  return %0 : f64
}
