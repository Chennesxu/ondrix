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

// A non-power-of-two window is the round_div consumer profile.
func.func @moving_average_odd_window(%input: tensor<12xi16>) -> tensor<10xi16> {
  // CHECK: ondrix.moving_average
  // CHECK-SAME: window = 3
  %result = ondrix.moving_average %input {
    window = 3 : i64,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>
  } : (tensor<12xi16>) -> tensor<10xi16>
  return %result : tensor<10xi16>
}

// An f32 DCT needs no rescaled output reading, so both numeric attributes
// name the same format.
// CHECK-LABEL: func.func @f32_dct
// CHECK: ondrix.dct
// CHECK-SAME: input_numeric = #ondsp.fp<format = f32, contract = fma>
func.func @f32_dct(%input: tensor<8xf32>) -> tensor<8xf32> {
  %result = ondrix.dct %input {
    input_numeric = #ondsp.fp<format = f32, contract = fma>,
    output_numeric = #ondsp.fp<format = f32, contract = fma>
  } : (tensor<8xf32>) -> tensor<8xf32>
  return %result : tensor<8xf32>
}

// CHECK-LABEL: func.func @f32_moving_average
// CHECK: ondrix.moving_average
// CHECK-SAME: numeric = #ondsp.fp<format = f32, contract = off>
func.func @f32_moving_average(%input: tensor<8xf32>) -> tensor<6xf32> {
  %result = ondrix.moving_average %input {
    window = 3 : i64,
    numeric = #ondsp.fp<format = f32, contract = off>
  } : (tensor<8xf32>) -> tensor<6xf32>
  return %result : tensor<6xf32>
}
