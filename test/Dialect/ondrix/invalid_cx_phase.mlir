// RUN: ondrix-opt %s --split-input-file --verify-diagnostics

// The result reading is an unsigned turn; a Q15 declaration would silently
// rescale every phase by half.
func.func @phase_with_the_q15_reading(%input: tensor<8xi32>) -> tensor<8xi16> {
  // expected-error @below {{cx_phase returns the unsigned Q0.16 or Q0.32 turn and must declare that reading}}
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

// -----

// The result storage follows the declared turn, not the component width.
func.func @phase_turn32_in_narrow_storage(%input: tensor<8xi32>) -> tensor<8xi16> {
  // expected-error @below {{executable phase requires tensor<Nxi32> to tensor<Nxi32> with static N in [1, 4096]}}
  %result = ondrix.cx_phase %input {
    layout = #ondsp.cx_layout<packed_i16_imag_hi_real_lo>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    output_numeric = #ondsp.fixed<unsigned, storage = i32, frac = 32>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<8xi32>) -> tensor<8xi16>
  return %result : tensor<8xi16>
}

// -----

// The f32 turn is the format's own reading; a fixed-point turn width beside
// it would name a storage the result does not have.
func.func @f32_declares_no_turn_width(%input: tensor<16xf32>) -> tensor<8xf32> {
  // expected-error @+1 {{the floating-point turn is the format's own and declares no output_numeric}}
  %out = ondrix.cx_phase %input {
    layout = #ondsp.cx_layout<interleaved>,
    numeric = #ondsp.fp<format = f32, contract = off>,
    output_numeric = #ondsp.fixed<unsigned, storage = i16, frac = 16>
  } : (tensor<16xf32>) -> tensor<8xf32>
  return %out : tensor<8xf32>
}

// -----

func.func @f32_has_no_boundary_to_round(%input: tensor<16xf32>) -> tensor<8xf32> {
  // expected-error @+1 {{floating-point phase rounds at no declared boundary of its own}}
  %out = ondrix.cx_phase %input {
    layout = #ondsp.cx_layout<interleaved>,
    numeric = #ondsp.fp<format = f32, contract = off>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<16xf32>) -> tensor<8xf32>
  return %out : tensor<8xf32>
}

// -----

func.func @f32_refuses_a_packed_layout(%input: tensor<16xf32>) -> tensor<8xf32> {
  // expected-error @+1 {{floating-point phase requires interleaved layout}}
  %out = ondrix.cx_phase %input {
    layout = #ondsp.cx_layout<packed_i16_imag_hi_real_lo>,
    numeric = #ondsp.fp<format = f32, contract = off>
  } : (tensor<16xf32>) -> tensor<8xf32>
  return %out : tensor<8xf32>
}

// -----

// The fixed-point profiles keep their reading: dropping the attribute is an
// undeclared turn width, not the f32 contract.
func.func @fixed_still_declares_its_turn(%input: tensor<8xi32>) -> tensor<8xi16> {
  // expected-error @+1 {{cx_phase returns the unsigned Q0.16 or Q0.32 turn and must declare that reading}}
  %out = ondrix.cx_phase %input {
    layout = #ondsp.cx_layout<packed_i16_imag_hi_real_lo>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<8xi32>) -> tensor<8xi16>
  return %out : tensor<8xi16>
}
