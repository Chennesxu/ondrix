// RUN: ondrix-compile %S/Inputs/q15_gain.ox | FileCheck %s
// RUN: ondrix-compile %S/Inputs/q15_gain_ties_positive.ox | FileCheck %s --check-prefix=TIES
// RUN: not ondrix-compile %S/Inputs/invalid_gain_constant.ox 2>&1 | FileCheck %s --check-prefix=CONSTANT
// RUN: not ondrix-compile %S/Inputs/invalid_gain_shape.ox 2>&1 | FileCheck %s --check-prefix=SHAPE
// RUN: not ondrix-compile %S/Inputs/invalid_gain_rounding.ox 2>&1 | FileCheck %s --check-prefix=ROUNDING
// RUN: ondrix-compile %S/Inputs/f32_gain.ox | FileCheck %s --check-prefix=FP
// RUN: not ondrix-compile %S/Inputs/invalid_f32_gain_denominator.ox 2>&1 | FileCheck %s --check-prefix=FPDEN
// RUN: not ondrix-compile %S/Inputs/invalid_f32_gain_magnitude.ox 2>&1 | FileCheck %s --check-prefix=FPMAG

// CHECK-LABEL: func.func @q15_gain(
// CHECK-SAME: %[[INPUT:.*]]: tensor<64xi16>) -> tensor<64xi16>
// CHECK: %[[RESULT:.*]] = ondrix.gain %[[INPUT]]
// CHECK-SAME: gain = 19661 : i64
// CHECK-SAME: numeric = #ondsp.fixed<signed, storage = i16, frac = 15>
// CHECK-SAME: rounding = #ondsp.rounding<nearest_even>
// CHECK: return %[[RESULT]] : tensor<64xi16>

// The binding exposes every tie rule the gain contract admits; omission
// above keeps the nearest_even default.
// TIES-LABEL: func.func @q15_gain_ties_positive(
// TIES: ondrix.gain
// TIES-SAME: gain = 19661 : i64
// TIES-SAME: rounding = #ondsp.rounding<nearest_ties_positive>

// CONSTANT: invalid_gain_constant.ox:2:10: error: gain constant must be a raw signed Q1.15 value in [-32768, 32767]
// SHAPE: invalid_gain_shape.ox:2:10: error: gain result extent must equal the input extent
// ROUNDING: invalid_gain_rounding.ox:2:10: error: gain rounding must be nearest_even or nearest_ties_positive

// FP-LABEL: func.func @f32_gain(
// FP: ondrix.gain
// FP-SAME: fp_gain = 3.750000e-01 : f32
// FP-SAME: numeric = #ondsp.fp<format = f32, contract = off>

// FPDEN: invalid_f32_gain_denominator.ox:2:10: error: f32 gain constant: denominator must be positive

// 548055821/548055723 is the witness that makes the wider bound wrong: its
// binary64 quotient lands exactly on a binary32 tie point and narrows one ulp
// above the correctly rounded value.
// FPMAG: invalid_f32_gain_magnitude.ox:2:10: error: f32 gain constant: numerator and denominator must not exceed 2^24
