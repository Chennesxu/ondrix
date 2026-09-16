// RUN: ondrix-opt %s --split-input-file --verify-diagnostics

func.func @widen_declares_a_policy(%x: tensor<8xi16>) -> tensor<8xi32> {
  // expected-error @below {{a widening conversion is exact and declares no rounding or overflow}}
  %0 = ondrix.quantize %x {
    src = #ondsp.fixed<signed, storage = i16, frac = 15>,
    dst = #ondsp.fixed<signed, storage = i32, frac = 31>,
    rounding = #ondsp.rounding<nearest_even>,
    overflow = #ondsp.overflow<saturate>
  } : (tensor<8xi16>) -> tensor<8xi32>
  return %0 : tensor<8xi32>
}

// -----

func.func @narrow_without_policy(%x: tensor<8xi32>) -> tensor<8xi16> {
  // expected-error @below {{a narrowing conversion requires rounding and overflow}}
  %0 = ondrix.quantize %x {
    src = #ondsp.fixed<signed, storage = i32, frac = 31>,
    dst = #ondsp.fixed<signed, storage = i16, frac = 15>
  } : (tensor<8xi32>) -> tensor<8xi16>
  return %0 : tensor<8xi16>
}

// -----

// Half a policy is no policy: the two attributes come and go together.
func.func @narrow_without_overflow(%x: tensor<8xi32>) -> tensor<8xi16> {
  // expected-error @below {{a narrowing conversion requires rounding and overflow}}
  %0 = ondrix.quantize %x {
    src = #ondsp.fixed<signed, storage = i32, frac = 31>,
    dst = #ondsp.fixed<signed, storage = i16, frac = 15>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<8xi32>) -> tensor<8xi16>
  return %0 : tensor<8xi16>
}

// -----

func.func @same_format(%x: tensor<8xi16>) -> tensor<8xi16> {
  // expected-error @below {{src and dst name the same format; a conversion changes the width}}
  %0 = ondrix.quantize %x {
    src = #ondsp.fixed<signed, storage = i16, frac = 15>,
    dst = #ondsp.fixed<signed, storage = i16, frac = 15>
  } : (tensor<8xi16>) -> tensor<8xi16>
  return %0 : tensor<8xi16>
}

// -----

func.func @accumulator_reading(%x: i32) -> i16 {
  // expected-error @below {{src and dst each require #ondsp.fixed<signed, storage = i16, frac = 15>, #ondsp.fixed<signed, storage = i32, frac = 31>, or #ondsp.fp<format = f32, contract = off>}}
  %0 = ondrix.quantize %x {
    src = #ondsp.fixed<signed, storage = i32, frac = 30>,
    dst = #ondsp.fixed<signed, storage = i16, frac = 15>,
    rounding = #ondsp.rounding<nearest_even>,
    overflow = #ondsp.overflow<saturate>
  } : (i32) -> i16
  return %0 : i16
}

// -----

// A dequantization's one rounding is IEEE's, not a declared choice.
func.func @dequantize_declares_a_policy(%x: tensor<8xi16>) -> tensor<8xf32> {
  // expected-error @below {{a dequantization is one IEEE rounding and declares no rounding or overflow}}
  %0 = ondrix.quantize %x {
    src = #ondsp.fixed<signed, storage = i16, frac = 15>,
    dst = #ondsp.fp<format = f32, contract = off>,
    rounding = #ondsp.rounding<nearest_even>,
    overflow = #ondsp.overflow<saturate>
  } : (tensor<8xi16>) -> tensor<8xf32>
  return %0 : tensor<8xf32>
}

// -----

func.func @quantize_without_policy(%x: tensor<8xf32>) -> tensor<8xi16> {
  // expected-error @below {{a quantization requires rounding and overflow}}
  %0 = ondrix.quantize %x {
    src = #ondsp.fp<format = f32, contract = off>,
    dst = #ondsp.fixed<signed, storage = i16, frac = 15>
  } : (tensor<8xf32>) -> tensor<8xi16>
  return %0 : tensor<8xi16>
}

// -----

// An unbounded real has no wrapping reading; the rail is the only policy.
func.func @quantize_wraps(%x: tensor<8xf32>) -> tensor<8xi16> {
  // expected-error @below {{a quantization saturates; wrapping an unbounded value is not a contract this operation offers}}
  %0 = ondrix.quantize %x {
    src = #ondsp.fp<format = f32, contract = off>,
    dst = #ondsp.fixed<signed, storage = i16, frac = 15>,
    rounding = #ondsp.rounding<nearest_even>,
    overflow = #ondsp.overflow<wrap>
  } : (tensor<8xf32>) -> tensor<8xi16>
  return %0 : tensor<8xi16>
}

// -----

func.func @float_to_float(%x: tensor<8xf32>) -> tensor<8xf32> {
  // expected-error @below {{a conversion between floating-point formats is not an operation here}}
  %0 = ondrix.quantize %x {
    src = #ondsp.fp<format = f32, contract = off>,
    dst = #ondsp.fp<format = f32, contract = off>
  } : (tensor<8xf32>) -> tensor<8xf32>
  return %0 : tensor<8xf32>
}

// -----

// The format's contract mode is off: a conversion contains no multiply-add.
func.func @contracting_format(%x: tensor<8xf32>) -> tensor<8xi16> {
  // expected-error @below {{src and dst each require #ondsp.fixed<signed, storage = i16, frac = 15>, #ondsp.fixed<signed, storage = i32, frac = 31>, or #ondsp.fp<format = f32, contract = off>}}
  %0 = ondrix.quantize %x {
    src = #ondsp.fp<format = f32, contract = fma>,
    dst = #ondsp.fixed<signed, storage = i16, frac = 15>,
    rounding = #ondsp.rounding<nearest_even>,
    overflow = #ondsp.overflow<saturate>
  } : (tensor<8xf32>) -> tensor<8xi16>
  return %0 : tensor<8xi16>
}

// -----

func.func @rank_two(%x: tensor<2x4xi16>) -> tensor<2x4xi32> {
  // expected-error @below {{executable conversions require a scalar or a static rank-1 tensor with N in [1, 4096]}}
  %0 = ondrix.quantize %x {
    src = #ondsp.fixed<signed, storage = i16, frac = 15>,
    dst = #ondsp.fixed<signed, storage = i32, frac = 31>
  } : (tensor<2x4xi16>) -> tensor<2x4xi32>
  return %0 : tensor<2x4xi32>
}

// -----

func.func @storage_mismatch(%x: tensor<8xi32>) -> tensor<8xi32> {
  // expected-error @below {{input element type must match source numeric storage type}}
  %0 = ondrix.quantize %x {
    src = #ondsp.fixed<signed, storage = i16, frac = 15>,
    dst = #ondsp.fixed<signed, storage = i32, frac = 31>
  } : (tensor<8xi32>) -> tensor<8xi32>
  return %0 : tensor<8xi32>
}
