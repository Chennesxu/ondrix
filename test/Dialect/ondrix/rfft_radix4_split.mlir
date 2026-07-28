// RUN: ondrix-opt %s | FileCheck %s

func.func @rfft32_radix4_split_q15(%input: tensor<32xi16>) -> tensor<17xi32> {
  // CHECK: ondrix.rfft_radix4_split
  // CHECK-SAME: input_numeric = #ondsp.fixed<signed, storage = i16, frac = 15>
  // CHECK-SAME: layout = #ondsp.cx_layout<packed_i16_imag_hi_real_lo>
  // CHECK-SAME: output_numeric = #ondsp.fixed<signed, storage = i16, frac = 10>
  // CHECK-SAME: product = #ondsp.product<full>
  %result = ondrix.rfft_radix4_split %input {
    layout = #ondsp.cx_layout<packed_i16_imag_hi_real_lo>,
    input_numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    output_numeric = #ondsp.fixed<signed, storage = i16, frac = 10>,
    product = #ondsp.product<full>
  } : (tensor<32xi16>) -> tensor<17xi32>
  return %result : tensor<17xi32>
}
