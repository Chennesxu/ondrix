// RUN: ondrix-opt %s | FileCheck %s

// The half-size split admission: N packed complex points in, N compact real
// bins out, with the Nyquist bin dropped by contract.

func.func @rfft_split4_q31(%input: tensor<4xi64>) -> tensor<4xi64> {
  // CHECK: ondrix.rfft_split
  // CHECK-SAME: input_numeric = #ondsp.fixed<signed, storage = i32, frac = 31>
  // CHECK-SAME: layout = #ondsp.cx_layout<packed_i32_imag_hi_real_lo>
  // CHECK-SAME: output_numeric = #ondsp.fixed<signed, storage = i32, frac = 31>
  // CHECK-SAME: product = #ondsp.product<full>
  %result = ondrix.rfft_split %input {
    layout = #ondsp.cx_layout<packed_i32_imag_hi_real_lo>,
    input_numeric = #ondsp.fixed<signed, storage = i32, frac = 31>,
    output_numeric = #ondsp.fixed<signed, storage = i32, frac = 31>,
    product = #ondsp.product<full>
  } : (tensor<4xi64>) -> tensor<4xi64>
  return %result : tensor<4xi64>
}

func.func @rfft_split32_q31(%input: tensor<32xi64>) -> tensor<32xi64> {
  // CHECK: ondrix.rfft_split
  %result = ondrix.rfft_split %input {
    layout = #ondsp.cx_layout<packed_i32_imag_hi_real_lo>,
    input_numeric = #ondsp.fixed<signed, storage = i32, frac = 31>,
    output_numeric = #ondsp.fixed<signed, storage = i32, frac = 31>,
    product = #ondsp.product<full>
  } : (tensor<32xi64>) -> tensor<32xi64>
  return %result : tensor<32xi64>
}
