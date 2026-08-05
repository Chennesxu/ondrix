// RUN: ondrix-opt %s -split-input-file --convert-ondsp-to-ortumcore -verify-diagnostics

// The readout realizes exactly toward_negative + saturate; every other
// export policy fails closed rather than selecting an unproven sequence.
func.func @rejects_nearest_even(%acc: !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>) -> i16 {
  // expected-error @+2 {{accumulator export is outside the proven readout capability}}
  // expected-error @+1 {{failed to legalize operation 'ondsp.acc_export'}}
  %0 = ondsp.acc_export %acc {dst = #ondsp.fixed<signed, storage = i16, frac = 15>, rounding = #ondsp.rounding<nearest_even>, overflow = #ondsp.overflow<saturate>} : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>) -> i16
  return %0 : i16
}

// -----

func.func @rejects_wrap_destination(%acc: !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>) -> i16 {
  // expected-error @+2 {{accumulator export is outside the proven readout capability}}
  // expected-error @+1 {{failed to legalize operation 'ondsp.acc_export'}}
  %0 = ondsp.acc_export %acc {dst = #ondsp.fixed<signed, storage = i16, frac = 15>, rounding = #ondsp.rounding<toward_negative>, overflow = #ondsp.overflow<wrap>} : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>) -> i16
  return %0 : i16
}

// -----

// frac 14 needs shift 16, one past the readout's mode-register range.
func.func @rejects_shift_beyond_range(%acc: !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>) -> i16 {
  // expected-error @+2 {{accumulator export is outside the proven readout capability}}
  // expected-error @+1 {{failed to legalize operation 'ondsp.acc_export'}}
  %0 = ondsp.acc_export %acc {dst = #ondsp.fixed<signed, storage = i16, frac = 14>, rounding = #ondsp.rounding<toward_negative>, overflow = #ondsp.overflow<saturate>} : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>) -> i16
  return %0 : i16
}
