// RUN: ondrix-opt %s --convert-ondrix-to-ondsp | FileCheck %s

// Each packed bin unpacks to signed halves, squares and sums exactly in
// i64, and routes the root through ondsp.sqrt_fixed; there is no other
// requantization boundary.

// CHECK-LABEL: func.func @cx_magnitude_q15
// CHECK-COUNT-5: ondsp.sqrt_fixed %{{.*}} {rounding = #ondsp.rounding<nearest_even>} : (i64) -> i16
// CHECK-NOT: ondsp.sqrt_fixed
// CHECK-NOT: ondrix.cx_magnitude

func.func @cx_magnitude_q15(%input: tensor<5xi32>) -> tensor<5xi16> {
  %magnitudes = ondrix.cx_magnitude %input {
    layout = #ondsp.cx_layout<packed_i16_imag_hi_real_lo>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<5xi32>) -> tensor<5xi16>
  return %magnitudes : tensor<5xi16>
}
