// RUN: ondrix-opt %s -split-input-file -verify-diagnostics

// The sum has no precision below its own fraction, so a reading above it
// would name bits the product does not have.
func.func @reading_cannot_exceed_the_product(%input: tensor<8xi32>) -> tensor<8xi32> {
  // expected-error @+1 {{output_numeric frac 31 exceeds the exact product frac 30}}
  %out = ondrix.cx_power %input {
    layout = #ondsp.cx_layout<packed_i16_imag_hi_real_lo>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    output_numeric = #ondsp.fixed<signed, storage = i32, frac = 31>,
    rounding = #ondsp.rounding<toward_negative>
  } : (tensor<8xi32>) -> tensor<8xi32>
  return %out : tensor<8xi32>
}

// -----

// Reading the sum at its own fraction shifts nothing, so declaring a tie rule
// there would name a boundary the program does not have.
func.func @no_shift_has_no_rounding(%input: tensor<8xi32>) -> tensor<8xi32> {
  // expected-error @+1 {{cx_power at this reading has no shift boundary to round}}
  %out = ondrix.cx_power %input {
    layout = #ondsp.cx_layout<packed_i16_imag_hi_real_lo>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    output_numeric = #ondsp.fixed<signed, storage = i32, frac = 30>,
    rounding = #ondsp.rounding<toward_negative>
  } : (tensor<8xi32>) -> tensor<8xi32>
  return %out : tensor<8xi32>
}

// -----

// The reverse: a reading below the sum's fraction is a shift, and a shift
// without a declared tie rule is an undeclared boundary.
func.func @a_shift_must_declare_its_rounding(%input: tensor<8xi32>) -> tensor<8xi16> {
  // expected-error @+1 {{shifts it right by 15 and must declare rounding}}
  %out = ondrix.cx_power %input {
    layout = #ondsp.cx_layout<packed_i16_imag_hi_real_lo>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    output_numeric = #ondsp.fixed<signed, storage = i16, frac = 15>
  } : (tensor<8xi32>) -> tensor<8xi16>
  return %out : tensor<8xi16>
}

// -----

// The Q31 profile would requantize each component before squaring, and
// whether restoring that costs anything depends on the declared reading.
func.func @q31_needs_its_own_evidence(%input: tensor<8xi64>) -> tensor<8xi32> {
  // expected-error @+1 {{executable squared magnitude requires packed_i16_imag_hi_real_lo layout}}
  %out = ondrix.cx_power %input {
    layout = #ondsp.cx_layout<packed_i32_imag_hi_real_lo>,
    numeric = #ondsp.fixed<signed, storage = i32, frac = 31>,
    output_numeric = #ondsp.fixed<signed, storage = i32, frac = 31>,
    rounding = #ondsp.rounding<toward_negative>
  } : (tensor<8xi64>) -> tensor<8xi32>
  return %out : tensor<8xi32>
}
