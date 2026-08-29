// RUN: ondrix-opt %s --convert-ondrix-to-ondsp | FileCheck %s
// RUN: ondrix-opt %s --convert-ondrix-to-ondsp | FileCheck %s --check-prefix=COEFF
// RUN: ondrix-opt %s --convert-ondrix-to-ondsp="sliding-window-reuse" | FileCheck %s --check-prefix=SLIDING

// The DCT8 lowering carries exact i64 products and sums; the only numeric
// boundaries are eight nearest-even round_shift exports by 16 + log2(8).
// Spot-checked generated coefficients (50-digit mpmath): c[1][0] = 32138,
// c[3][2] = -32138, c[7][7] = -6393; the k = 0 row saturates to 32767.

// COEFF-LABEL: func.func @dct8_q15
// COEFF-DAG: arith.constant 32138 : i64
// COEFF-DAG: arith.constant -32138 : i64
// COEFF-DAG: arith.constant -6393 : i64
// COEFF-DAG: arith.constant 32767 : i64

// CHECK-LABEL: func.func @dct8_q15
// CHECK-COUNT-8: ondsp.round_shift {{.*}}post_shift_right = 19, rounding = nearest_even, overflow = saturate, saturate_to = i16{{.*}} (i64) -> i16
// CHECK-NOT: ondsp.round_shift
// CHECK-NOT: ondrix.dct

func.func @dct8_q15(%input: tensor<8xi16>) -> tensor<8xi16> {
  %result = ondrix.dct %input {
    input_numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    output_numeric = #ondsp.fixed<signed, storage = i16, frac = 11>
  } : (tensor<8xi16>) -> tensor<8xi16>
  return %result : tensor<8xi16>
}

// The sliding mode reuses the exact running sum: one initial window of
// seven adds, then one add and one subtract per further output. The
// rounding boundaries are unchanged.

// SLIDING-LABEL: func.func @moving_average_q15
// SLIDING-COUNT-4: arith.subi
// SLIDING-NOT: arith.subi
// SLIDING: ondsp.round_shift
// SLIDING-NOT: ondrix.moving_average

// CHECK-LABEL: func.func @moving_average_q15
// CHECK-COUNT-5: ondsp.round_shift {{.*}}post_shift_right = 3, rounding = nearest_even{{.*}} (i64) -> i16
// CHECK-NOT: ondsp.round_shift
// CHECK-NOT: ondrix.moving_average

func.func @moving_average_q15(%input: tensor<12xi16>) -> tensor<5xi16> {
  %result = ondrix.moving_average %input {
    window = 8 : i64,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>
  } : (tensor<12xi16>) -> tensor<5xi16>
  return %result : tensor<5xi16>
}

// A general window keeps the exact i64 sums but its single boundary is the
// nearest-even round_div by K; the power-of-two profile above must keep its
// round_shift spelling untouched.

// CHECK-LABEL: func.func @moving_average_odd_q15
// CHECK-COUNT-10: ondsp.round_div {{.*}}divisor = 3 : i64, overflow = #ondsp.overflow<saturate>, pre_shift_left = 0 : i64, rounding = #ondsp.rounding<nearest_even>{{.*}} (i64) -> i16
// CHECK-NOT: ondsp.round_div
// CHECK-NOT: ondsp.round_shift
// CHECK-NOT: ondrix.moving_average

func.func @moving_average_odd_q15(%input: tensor<12xi16>) -> tensor<10xi16> {
  %result = ondrix.moving_average %input {
    window = 3 : i64,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>
  } : (tensor<12xi16>) -> tensor<10xi16>
  return %result : tensor<10xi16>
}

// The declared floor boundary rides the same single round_shift per output.
// CHECK-LABEL: func.func @dct8_floor_q15
// CHECK-COUNT-8: ondsp.round_shift {{.*}}post_shift_right = 19, rounding = toward_negative, overflow = saturate, saturate_to = i16{{.*}} (i64) -> i16
// CHECK-NOT: ondsp.round_shift
func.func @dct8_floor_q15(%input: tensor<8xi16>) -> tensor<8xi16> {
  %result = ondrix.dct %input {
    input_numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    output_numeric = #ondsp.fixed<signed, storage = i16, frac = 11>,
    rounding = #ondsp.rounding<toward_negative>
  } : (tensor<8xi16>) -> tensor<8xi16>
  return %result : tensor<8xi16>
}
