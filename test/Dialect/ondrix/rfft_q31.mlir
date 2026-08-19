// RUN: ondrix-opt %s | FileCheck %s

// The packed-Q31 real-spectrum admission: real i32 samples to i64 Hermitian
// bins under the re-frozen raw-high stage policy.
func.func @rfft8_q31(%input: tensor<8xi32>) -> tensor<5xi64> {
  // CHECK: ondrix.rfft
  // CHECK-SAME: layout = #ondsp.cx_layout<packed_i32_imag_hi_real_lo>
  // CHECK-SAME: numeric = #ondsp.fixed<signed, storage = i32, frac = 31>
  // CHECK-SAME: output_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 1, rounding = toward_negative, overflow = saturate, saturate_to = i32>
  // CHECK-SAME: product = #ondsp.product<high_raw>
  // CHECK-SAME: product_scale = #ondsp.scale<pre_shift_left = 1, post_shift_right = 0, rounding = toward_negative, overflow = saturate, saturate_to = i32>
  %result = ondrix.rfft %input {
    layout = #ondsp.cx_layout<packed_i32_imag_hi_real_lo>,
    numeric = #ondsp.fixed<signed, storage = i32, frac = 31>,
    product = #ondsp.product<high_raw>,
    product_scale = #ondsp.scale<pre_shift_left = 1, post_shift_right = 0, rounding = toward_negative, overflow = saturate, saturate_to = i32>,
    output_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 1, rounding = toward_negative, overflow = saturate, saturate_to = i32>
  } : (tensor<8xi32>) -> tensor<5xi64>
  return %result : tensor<5xi64>
}

func.func @irfft64_q31(%input: tensor<33xi64>) -> tensor<64xi32> {
  // CHECK: ondrix.irfft
  %result = ondrix.irfft %input {
    layout = #ondsp.cx_layout<packed_i32_imag_hi_real_lo>,
    numeric = #ondsp.fixed<signed, storage = i32, frac = 31>,
    product = #ondsp.product<high_raw>,
    product_scale = #ondsp.scale<pre_shift_left = 1, post_shift_right = 0, rounding = toward_negative, overflow = saturate, saturate_to = i32>,
    output_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 1, rounding = toward_negative, overflow = saturate, saturate_to = i32>
  } : (tensor<33xi64>) -> tensor<64xi32>
  return %result : tensor<64xi32>
}
