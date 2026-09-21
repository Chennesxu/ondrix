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

// -----

// An f32 sum names its own reading; a fixed-point one beside it would name a
// boundary the program does not have.
func.func @f32_declares_no_reading(%input: tensor<16xf32>) -> tensor<8xf32> {
  // expected-error @+1 {{floating-point squared magnitude names its own reading and declares no output_numeric}}
  %out = ondrix.cx_power %input {
    layout = #ondsp.cx_layout<interleaved>,
    numeric = #ondsp.fp<format = f32, contract = off>,
    output_numeric = #ondsp.fixed<signed, storage = i16, frac = 15>
  } : (tensor<16xf32>) -> tensor<8xf32>
  return %out : tensor<8xf32>
}

// -----

func.func @f32_has_no_boundary_to_round(%input: tensor<16xf32>) -> tensor<8xf32> {
  // expected-error @+1 {{floating-point squared magnitude rounds at no declared boundary of its own}}
  %out = ondrix.cx_power %input {
    layout = #ondsp.cx_layout<interleaved>,
    numeric = #ondsp.fp<format = f32, contract = off>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<16xf32>) -> tensor<8xf32>
  return %out : tensor<8xf32>
}

// -----

// The packed layouts name a container; the f32 profile has none, so its
// complex value is two adjacent elements and only interleaved says that.
func.func @f32_refuses_a_packed_layout(%input: tensor<16xf32>) -> tensor<8xf32> {
  // expected-error @+1 {{floating-point squared magnitude requires interleaved layout}}
  %out = ondrix.cx_power %input {
    layout = #ondsp.cx_layout<packed_i16_imag_hi_real_lo>,
    numeric = #ondsp.fp<format = f32, contract = off>
  } : (tensor<16xf32>) -> tensor<8xf32>
  return %out : tensor<8xf32>
}

// -----

// A bin is two elements, so an element-for-element result would read half the
// operand as a whole spectrum.
func.func @f32_reads_two_elements_per_bin(%input: tensor<8xf32>) -> tensor<8xf32> {
  // expected-error @+1 {{executable floating-point squared magnitude requires tensor<2Nxf32> to tensor<Nxf32> with static N in [1, 4096]}}
  %out = ondrix.cx_power %input {
    layout = #ondsp.cx_layout<interleaved>,
    numeric = #ondsp.fp<format = f32, contract = off>
  } : (tensor<8xf32>) -> tensor<8xf32>
  return %out : tensor<8xf32>
}
