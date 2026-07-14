// RUN: ondrix-opt %s -split-input-file -verify-diagnostics

func.func @fir_requires_rank_one(
    %input: memref<2x4xi16>, %coeffs: memref<2x4xi16>) -> i32 {
  // expected-error@+1 {{requires rank-1 input and coefficient windows}}
  %0 = ondrix.fir %input, %coeffs {numeric = #ondsp.fixed<signed, storage = i16, frac = 15>, product = #ondsp.product<full>} : (memref<2x4xi16>, memref<2x4xi16>) -> i32
  return %0 : i32
}

// -----

func.func @fir_requires_equal_static_lengths(
    %input: memref<8xi16>, %coeffs: memref<4xi16>) -> i32 {
  // expected-error@+1 {{input and coefficient windows must have equal length}}
  %0 = ondrix.fir %input, %coeffs {numeric = #ondsp.fixed<signed, storage = i16, frac = 15>, product = #ondsp.product<full>} : (memref<8xi16>, memref<4xi16>) -> i32
  return %0 : i32
}

// -----

func.func @fir_requires_matching_elements(
    %input: memref<8xi16>, %coeffs: memref<8xi32>) -> i32 {
  // expected-error@+1 {{input and coefficient element types must match}}
  %0 = ondrix.fir %input, %coeffs {numeric = #ondsp.fixed<signed, storage = i16, frac = 15>, product = #ondsp.product<full>} : (memref<8xi16>, memref<8xi32>) -> i32
  return %0 : i32
}

// -----

func.func @fir_requires_scalar_result(
    %input: memref<8xi16>, %coeffs: memref<8xi16>) -> tensor<1xi32> {
  // expected-error@+1 {{requires a scalar integer or floating-point result}}
  %0 = ondrix.fir %input, %coeffs {numeric = #ondsp.fixed<signed, storage = i16, frac = 15>, product = #ondsp.product<full>} : (memref<8xi16>, memref<8xi16>) -> tensor<1xi32>
  return %0 : tensor<1xi32>
}

// -----

func.func @fir_requires_numeric_storage(
    %input: memref<8xi32>, %coeffs: memref<8xi32>) -> i64 {
  // expected-error@+1 {{window element type must match fixed numeric storage type}}
  %0 = ondrix.fir %input, %coeffs {numeric = #ondsp.fixed<signed, storage = i16, frac = 15>, product = #ondsp.product<full>} : (memref<8xi32>, memref<8xi32>) -> i64
  return %0 : i64
}

// -----

func.func @fir_requires_fp_result_format(
    %input: memref<8xf32>, %coeffs: memref<8xf32>) -> f64 {
  // expected-error@+1 {{floating-point FIR window and result types must match numeric format}}
  %0 = ondrix.fir %input, %coeffs {numeric = #ondsp.fp<format = f32, contract = fma>} : (memref<8xf32>, memref<8xf32>) -> f64
  return %0 : f64
}

// -----

func.func @fir_accepts_dynamic_windows(
    %input: memref<?xf32>, %coeffs: memref<?xf32>) -> f32 {
  %0 = ondrix.fir %input, %coeffs {numeric = #ondsp.fp<format = f32, contract = fma>} : (memref<?xf32>, memref<?xf32>) -> f32
  return %0 : f32
}
