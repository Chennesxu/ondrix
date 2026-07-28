// RUN: ondrix-opt %s --convert-ondrix-to-ondsp | FileCheck %s

// Spot-checks of the generated packed Q15 twiddle constants for the size-32
// combine stage against 50-digit mpmath references:
//   k=1:  (cos, -sin)(2*pi/32)   = (32138, -6393)  -> 0xe7077d8a = -418939510
//   k=5:  (cos, -sin)(10*pi/32)  = (18205, -27246) -> 0x9592471d = -1785575651
//   k=12: (cos, -sin)(24*pi/32)  = (-23170,-23170) -> 0xa57ea57e = -1518426754

// CHECK-LABEL: func.func @cfft32_q15
// CHECK-DAG: arith.constant -418939510 : i32
// CHECK-DAG: arith.constant -1785575651 : i32
// CHECK-DAG: arith.constant -1518426754 : i32
// CHECK-NOT: ondrix.cfft

func.func @cfft32_q15(%input: tensor<32xi32>) -> tensor<32xi32> {
  %result = ondrix.cfft %input {
    direction = #ondrix.cfft_direction<forward>,
    layout = #ondsp.cx_layout<packed_i16_imag_hi_real_lo>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    product = #ondsp.product<full>,
    product_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 15, rounding = nearest_even, overflow = saturate, saturate_to = i16>,
    output_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 1, rounding = nearest_even, overflow = saturate, saturate_to = i16>
  } : (tensor<32xi32>) -> tensor<32xi32>
  return %result : tensor<32xi32>
}
