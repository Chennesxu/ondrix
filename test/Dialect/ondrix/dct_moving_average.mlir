// RUN: ondrix-opt %s | FileCheck %s

func.func @dct8(%input: tensor<8xi16>) -> tensor<8xi16> {
  // CHECK: ondrix.dct
  // CHECK-SAME: input_numeric = #ondsp.fixed<signed, storage = i16, frac = 15>
  // CHECK-SAME: output_numeric = #ondsp.fixed<signed, storage = i16, frac = 11>
  %result = ondrix.dct %input {
    input_numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    output_numeric = #ondsp.fixed<signed, storage = i16, frac = 11>
  } : (tensor<8xi16>) -> tensor<8xi16>
  return %result : tensor<8xi16>
}

func.func @dct64(%input: tensor<64xi16>) -> tensor<64xi16> {
  // CHECK: ondrix.dct
  // CHECK-SAME: frac = 8>
  %result = ondrix.dct %input {
    input_numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    output_numeric = #ondsp.fixed<signed, storage = i16, frac = 8>
  } : (tensor<64xi16>) -> tensor<64xi16>
  return %result : tensor<64xi16>
}

func.func @moving_average(%input: tensor<40xi16>) -> tensor<33xi16> {
  // CHECK: ondrix.moving_average
  // CHECK-SAME: window = 8
  %result = ondrix.moving_average %input {
    window = 8 : i64,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>
  } : (tensor<40xi16>) -> tensor<33xi16>
  return %result : tensor<33xi16>
}
