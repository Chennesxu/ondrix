// RUN: ondrix-opt %s | FileCheck %s

func.func @cx_magnitude(%input: tensor<33xi32>) -> tensor<33xi16> {
  // CHECK: ondrix.cx_magnitude
  // CHECK-SAME: layout = #ondsp.cx_layout<packed_i16_imag_hi_real_lo>
  // CHECK-SAME: numeric = #ondsp.fixed<signed, storage = i16, frac = 15>
  // CHECK-SAME: rounding = #ondsp.rounding<nearest_even>
  %magnitudes = ondrix.cx_magnitude %input {
    layout = #ondsp.cx_layout<packed_i16_imag_hi_real_lo>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<33xi32>) -> tensor<33xi16>
  return %magnitudes : tensor<33xi16>
}

func.func @cx_magnitude_floor(%input: tensor<5xi32>) -> tensor<5xi16> {
  // CHECK: ondrix.cx_magnitude
  // CHECK-SAME: rounding = #ondsp.rounding<toward_negative>
  %magnitudes = ondrix.cx_magnitude %input {
    layout = #ondsp.cx_layout<packed_i16_imag_hi_real_lo>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    rounding = #ondsp.rounding<toward_negative>
  } : (tensor<5xi32>) -> tensor<5xi16>
  return %magnitudes : tensor<5xi16>
}

func.func @sqrt_fixed(%input: i64) -> i16 {
  // CHECK: ondsp.sqrt_fixed
  // CHECK-SAME: rounding = #ondsp.rounding<nearest_even>
  %root = ondsp.sqrt_fixed %input {
    rounding = #ondsp.rounding<nearest_even>
  } : (i64) -> i16
  return %root : i16
}
