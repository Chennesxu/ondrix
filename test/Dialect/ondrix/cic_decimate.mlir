// RUN: ondrix-opt %s | ondrix-opt | FileCheck %s

// CHECK-LABEL: func.func @cic_two_stage_wrap
// CHECK: ondrix.cic_decimate
// CHECK-SAME: delay = 1
// CHECK-SAME: overflow = #ondsp.overflow<wrap>
// CHECK-SAME: rate = 4
// CHECK-SAME: stages = 2
func.func @cic_two_stage_wrap(%input: tensor<32xi16>) -> tensor<8xi16> {
  %result = ondrix.cic_decimate %input {
    stages = 2 : i64,
    rate = 4 : i64,
    delay = 1 : i64,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    overflow = #ondsp.overflow<wrap>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<32xi16>) -> tensor<8xi16>
  return %result : tensor<8xi16>
}

// The contract admits saturate so the two programs stay comparable; only
// wrap implements the cascade.
// CHECK-LABEL: func.func @cic_saturating_state
// CHECK: ondrix.cic_decimate
// CHECK-SAME: overflow = #ondsp.overflow<saturate>
func.func @cic_saturating_state(%input: tensor<32xi16>) -> tensor<8xi16> {
  %result = ondrix.cic_decimate %input {
    stages = 2 : i64,
    rate = 4 : i64,
    delay = 1 : i64,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    overflow = #ondsp.overflow<saturate>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<32xi16>) -> tensor<8xi16>
  return %result : tensor<8xi16>
}

// CHECK-LABEL: func.func @cic_widest_admissible_growth
// CHECK: ondrix.cic_decimate
// CHECK-SAME: rate = 4096
// CHECK-SAME: stages = 4
func.func @cic_widest_admissible_growth(%input: tensor<4096xi16>) -> tensor<1xi16> {
  %result = ondrix.cic_decimate %input {
    stages = 4 : i64,
    rate = 4096 : i64,
    delay = 1 : i64,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    overflow = #ondsp.overflow<wrap>,
    rounding = #ondsp.rounding<nearest_ties_positive>
  } : (tensor<4096xi16>) -> tensor<1xi16>
  return %result : tensor<1xi16>
}

// The Q31 profile keeps the same cascade and shifts the register to W + G;
// only the growth budget narrows.
// CHECK-LABEL: func.func @q31_cic_decimate
// CHECK: ondrix.cic_decimate
// CHECK-SAME: numeric = #ondsp.fixed<signed, storage = i32, frac = 31>
func.func @q31_cic_decimate(%input: tensor<256xi32>) -> tensor<4xi32> {
  %result = ondrix.cic_decimate %input {
    stages = 4 : i64, rate = 64 : i64, delay = 1 : i64,
    numeric = #ondsp.fixed<signed, storage = i32, frac = 31>,
    overflow = #ondsp.overflow<wrap>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<256xi32>) -> tensor<4xi32>
  return %result : tensor<4xi32>
}
