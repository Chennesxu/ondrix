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
  // expected-error @below {{src and dst require #ondsp.fixed<signed, storage = i16, frac = 15> or #ondsp.fixed<signed, storage = i32, frac = 31>}}
  %0 = ondrix.quantize %x {
    src = #ondsp.fixed<signed, storage = i32, frac = 30>,
    dst = #ondsp.fixed<signed, storage = i16, frac = 15>,
    rounding = #ondsp.rounding<nearest_even>,
    overflow = #ondsp.overflow<saturate>
  } : (i32) -> i16
  return %0 : i16
}

// -----

func.func @floating_point_destination(%x: tensor<8xi16>) -> tensor<8xf32> {
  // expected-error @below {{src and dst require #ondsp.fixed<signed, storage = i16, frac = 15> or #ondsp.fixed<signed, storage = i32, frac = 31>}}
  %0 = ondrix.quantize %x {
    src = #ondsp.fixed<signed, storage = i16, frac = 15>,
    dst = #ondsp.fp<format = f32, contract = off>
  } : (tensor<8xi16>) -> tensor<8xf32>
  return %0 : tensor<8xf32>
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
