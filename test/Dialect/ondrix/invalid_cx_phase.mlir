// RUN: ondrix-opt %s --split-input-file --verify-diagnostics

// The result reading is the unsigned Q0.16 turn; a Q15 declaration would
// silently rescale every phase by half.
func.func @phase_with_the_q15_reading(%input: tensor<8xi32>) -> tensor<8xi16> {
  // expected-error @below {{cx_phase returns the unsigned Q0.16 turn and must declare that reading}}
  %result = ondrix.cx_phase %input {
    layout = #ondsp.cx_layout<packed_i16_imag_hi_real_lo>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    output_numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<8xi32>) -> tensor<8xi16>
  return %result : tensor<8xi16>
}

// -----

func.func @phase_with_a_directed_rule(%input: tensor<8xi32>) -> tensor<8xi16> {
  // expected-error @below {{cx_phase requires nearest_even rounding}}
  %result = ondrix.cx_phase %input {
    layout = #ondsp.cx_layout<packed_i16_imag_hi_real_lo>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    output_numeric = #ondsp.fixed<unsigned, storage = i16, frac = 16>,
    rounding = #ondsp.rounding<toward_negative>
  } : (tensor<8xi32>) -> tensor<8xi16>
  return %result : tensor<8xi16>
}

// -----

func.func @phase_with_a_dynamic_extent(%input: tensor<?xi32>) -> tensor<?xi16> {
  // expected-error @below {{executable phase requires tensor<Nxi32> to tensor<Nxi16> with static N in [1, 4096]}}
  %result = ondrix.cx_phase %input {
    layout = #ondsp.cx_layout<packed_i16_imag_hi_real_lo>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    output_numeric = #ondsp.fixed<unsigned, storage = i16, frac = 16>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<?xi32>) -> tensor<?xi16>
  return %result : tensor<?xi16>
}

// -----

func.func @phase_with_mismatched_extents(%input: tensor<8xi32>) -> tensor<4xi16> {
  // expected-error @below {{executable phase requires tensor<Nxi32> to tensor<Nxi16> with static N in [1, 4096]}}
  %result = ondrix.cx_phase %input {
    layout = #ondsp.cx_layout<packed_i16_imag_hi_real_lo>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    output_numeric = #ondsp.fixed<unsigned, storage = i16, frac = 16>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<8xi32>) -> tensor<4xi16>
  return %result : tensor<4xi16>
}
