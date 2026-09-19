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
