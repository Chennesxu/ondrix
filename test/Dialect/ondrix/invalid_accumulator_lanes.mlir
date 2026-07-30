// RUN: ondrix-opt %s -split-input-file -verify-diagnostics

// Algorithm intent never declares a lane count. Batching outputs across lanes
// is a decision an explicit lowering pass makes after the schedule is known, so
// a multi-lane accumulator in an intent attribute or an intent result would be
// an unproven claim about the schedule rather than a numeric contract. Every
// ondrix operation that names an accumulator refuses it.

func.func @fir_decimate_rejects_lanes(
    %input: tensor<44xi16>, %coeffs: tensor<8xi16>, %init: tensor<19xi16>) -> tensor<19xi16> {
  // expected-error@+1 {{algorithm accumulator must be single-lane; lane batching is a lowering decision, not part of the algorithm contract}}
  %result = ondrix.fir_decimate %input, %coeffs, %init {
    accumulator = !ondsp.acc<storage = i40, frac = 30, signed,
                              update_overflow = saturate, lanes = 8>,
    dst = #ondsp.fixed<signed, storage = i16, frac = 15>,
    factor = 2,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    overflow = #ondsp.overflow<saturate>,
    product = #ondsp.product<full>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<44xi16>, tensor<8xi16>, tensor<19xi16>) -> tensor<19xi16>
  return %result : tensor<19xi16>
}

// -----

func.func @fir_filter_rejects_lanes(
    %input: tensor<12xi16>, %coeffs: tensor<5xi16>, %init: tensor<8xi16>) -> tensor<8xi16> {
  // expected-error@+1 {{algorithm accumulator must be single-lane; lane batching is a lowering decision, not part of the algorithm contract}}
  %result = ondrix.fir_filter %input, %coeffs, %init {
    accumulator = !ondsp.acc<storage = i40, frac = 30, signed,
                              update_overflow = saturate, lanes = 8>,
    boundary = #ondrix.fir_boundary<valid>,
    dst = #ondsp.fixed<signed, storage = i16, frac = 15>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    overflow = #ondsp.overflow<saturate>,
    product = #ondsp.product<full>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<12xi16>, tensor<5xi16>, tensor<8xi16>) -> tensor<8xi16>
  return %result : tensor<8xi16>
}

// -----

// The accumulator also appears as an intent RESULT type, which is a second
// presence-only surface the lane parameter would otherwise widen.
func.func @fir_rejects_lanes(%input: memref<8xi16>, %coeffs: memref<8xi16>)
    -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate, lanes = 8> {
  // expected-error@+1 {{algorithm accumulator must be single-lane; lane batching is a lowering decision, not part of the algorithm contract}}
  %result = ondrix.fir %input, %coeffs {
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    product = #ondsp.product<full>
  } : (memref<8xi16>, memref<8xi16>)
      -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate, lanes = 8>
  return %result
      : !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate, lanes = 8>
}

// -----

func.func @dot_rejects_lanes(%lhs: memref<8xi16>, %rhs: memref<8xi16>)
    -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate, lanes = 8> {
  // expected-error@+1 {{algorithm accumulator must be single-lane; lane batching is a lowering decision, not part of the algorithm contract}}
  %result = ondrix.dot %lhs, %rhs {
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    product = #ondsp.product<full>
  } : (memref<8xi16>, memref<8xi16>)
      -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate, lanes = 8>
  return %result
      : !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate, lanes = 8>
}

// -----

// The direct-form-II section validates its accumulator against the executable
// width profiles directly rather than through the shared reduction check, so it
// carries its own refusal.
func.func @sos_filter_df2_fixed_rejects_lanes(
    %input: tensor<8xi16>, %coeffs: tensor<1x5xi16>, %scales: tensor<1xi16>,
    %state: tensor<1x2xi16>) -> (tensor<8xi16>, tensor<1x2xi16>) {
  // expected-error@+1 {{algorithm accumulator must be single-lane; lane batching is a lowering decision, not part of the algorithm contract}}
  %output, %next = ondrix.sos_filter_df2_fixed %input, %coeffs, %scales, %state {
    accumulator = !ondsp.acc<storage = i40, frac = 30, signed,
                              update_overflow = saturate, lanes = 8>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    output_overflow = #ondsp.overflow<saturate>,
    output_rounding = #ondsp.rounding<nearest_even>,
    product = #ondsp.product<full>,
    state_overflow = #ondsp.overflow<saturate>,
    state_rounding = #ondsp.rounding<nearest_even>
  } : (tensor<8xi16>, tensor<1x5xi16>, tensor<1xi16>, tensor<1x2xi16>)
      -> (tensor<8xi16>, tensor<1x2xi16>)
  return %output, %next : tensor<8xi16>, tensor<1x2xi16>
}
