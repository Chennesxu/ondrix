// RUN: ondrix-opt %s -split-input-file -verify-diagnostics

!acc = !ondsp.acc<storage = i32, frac = 30, signed, update_overflow = saturate>

// The valid boundary fixes the output length, so a destination of another
// length would silently leave samples unwritten.
func.func @output_length_follows_the_window(
    %input: tensor<10xi32>, %coeffs: tensor<4xi32>, %init: tensor<10xi32>) -> tensor<10xi32> {
  // expected-error @+1 {{valid boundary output length must be 7, not 10}}
  %out = ondrix.cx_fir_filter %input, %coeffs, %init {
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    layout = #ondsp.cx_layout<packed_i16_imag_hi_real_lo>,
    accumulator = !acc,
    rounding = #ondsp.rounding<toward_negative>,
    overflow = #ondsp.overflow<saturate>
  } : (tensor<10xi32>, tensor<4xi32>, tensor<10xi32>) -> tensor<10xi32>
  return %out : tensor<10xi32>
}

// -----

!acc = !ondsp.acc<storage = i32, frac = 15, signed, update_overflow = saturate>

// The per-sample contract is the reduction's, so a requantizing accumulator
// is refused at the window level too.
func.func @accumulator_frac_must_be_exact(
    %input: tensor<10xi32>, %coeffs: tensor<4xi32>, %init: tensor<7xi32>) -> tensor<7xi32> {
  // expected-error @+1 {{accumulator frac 15 does not match the exact product frac 30}}
  %out = ondrix.cx_fir_filter %input, %coeffs, %init {
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    layout = #ondsp.cx_layout<packed_i16_imag_hi_real_lo>,
    accumulator = !acc,
    rounding = #ondsp.rounding<toward_negative>,
    overflow = #ondsp.overflow<saturate>
  } : (tensor<10xi32>, tensor<4xi32>, tensor<7xi32>) -> tensor<7xi32>
  return %out : tensor<7xi32>
}

// -----

!acc = !ondsp.acc<storage = i32, frac = 30, signed, update_overflow = saturate>

// A component-width element would halve the term count without saying so.
func.func @operand_must_be_the_packed_container(
    %input: tensor<10xi16>, %coeffs: tensor<4xi16>, %init: tensor<7xi16>) -> tensor<7xi16> {
  // expected-error @+1 {{packed operand element must be i32 for this layout}}
  %out = ondrix.cx_fir_filter %input, %coeffs, %init {
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    layout = #ondsp.cx_layout<packed_i16_imag_hi_real_lo>,
    accumulator = !acc,
    rounding = #ondsp.rounding<toward_negative>,
    overflow = #ondsp.overflow<saturate>
  } : (tensor<10xi16>, tensor<4xi16>, tensor<7xi16>) -> tensor<7xi16>
  return %out : tensor<7xi16>
}

// -----

// The format is the carrier at this profile, so the fixed triple names
// boundaries the program does not have.
func.func @f32_declares_no_triple(%input: tensor<16xf32>, %coeffs: tensor<6xf32>,
                                  %init: tensor<12xf32>) -> tensor<12xf32> {
  // expected-error @+1 {{floating-point cx_fir_filter accumulates in the numeric format and declares no accumulator, rounding, or overflow}}
  %r = ondrix.cx_fir_filter %input, %coeffs, %init {
    numeric = #ondsp.fp<format = f32, contract = off>,
    layout = #ondsp.cx_layout<interleaved>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<16xf32>, tensor<6xf32>, tensor<12xf32>) -> tensor<12xf32>
  return %r : tensor<12xf32>
}

// -----

func.func @f32_reads_two_elements_per_value(%input: tensor<15xf32>, %coeffs: tensor<6xf32>,
                                            %init: tensor<11xf32>) -> tensor<11xf32> {
  // expected-error @+1 {{floating-point cx_fir_filter reads two elements per complex value and requires even tensor lengths}}
  %r = ondrix.cx_fir_filter %input, %coeffs, %init {
    numeric = #ondsp.fp<format = f32, contract = off>,
    layout = #ondsp.cx_layout<interleaved>
  } : (tensor<15xf32>, tensor<6xf32>, tensor<11xf32>) -> tensor<11xf32>
  return %r : tensor<11xf32>
}

// -----

// The window arithmetic counts complex values, so an element-counted result
// is off by the factor the layout declares.
func.func @f32_valid_boundary_counts_values(%input: tensor<16xf32>, %coeffs: tensor<6xf32>,
                                            %init: tensor<22xf32>) -> tensor<22xf32> {
  // expected-error @+1 {{valid boundary output length must be 6, not 11}}
  %r = ondrix.cx_fir_filter %input, %coeffs, %init {
    numeric = #ondsp.fp<format = f32, contract = off>,
    layout = #ondsp.cx_layout<interleaved>
  } : (tensor<16xf32>, tensor<6xf32>, tensor<22xf32>) -> tensor<22xf32>
  return %r : tensor<22xf32>
}

// -----

// Dropping the triple at a packed width is an undeclared boundary, not the
// floating-point contract.
func.func @packed_still_declares_its_triple(%input: tensor<10xi32>, %coeffs: tensor<4xi32>,
                                            %init: tensor<7xi32>) -> tensor<7xi32> {
  // expected-error @+1 {{cx_fir_filter requires an accumulator, rounding, and overflow at this numeric policy}}
  %r = ondrix.cx_fir_filter %input, %coeffs, %init {
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    layout = #ondsp.cx_layout<packed_i16_imag_hi_real_lo>
  } : (tensor<10xi32>, tensor<4xi32>, tensor<7xi32>) -> tensor<7xi32>
  return %r : tensor<7xi32>
}
