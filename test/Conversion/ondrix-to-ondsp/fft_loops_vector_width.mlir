// RUN: ondrix-opt %s --convert-ondrix-to-ondsp="fft-loops fft-loops-vector-width=4" | FileCheck %s
// RUN: not ondrix-opt %s --convert-ondrix-to-ondsp="fft-loops fft-loops-vector-width=6" 2>&1 | FileCheck %s --check-prefix=ODD

// Stages below the lane count stay scalar; from half length 4 on, each phase
// step reads four pairs and four twiddles as vectors and writes both halves back.
// CHECK-LABEL: func.func @cfft64_forward_q15
// CHECK: scf.for %{{.*}} = %c0 to %c2 step %c1
// CHECK: ondsp.cx_butterfly {{.*}} : (i32, i32, i32) -> (i32, i32)
// CHECK: scf.for %{{.*}} = %c2 to %c6 step %c1
// CHECK: scf.for %{{.*}} = %c0 to %{{.*}} step %c4
// CHECK: vector.transfer_read {{.*}} : tensor<64xi32>, vector<4xi32>
// CHECK: vector.transfer_read {{.*}} : tensor<64xi32>, vector<4xi32>
// CHECK: vector.transfer_read {{.*}} : tensor<64xi32>, vector<4xi32>
// CHECK: ondsp.cx_butterfly {{.*}} : (vector<4xi32>, vector<4xi32>, vector<4xi32>) -> (vector<4xi32>, vector<4xi32>)
// CHECK: vector.transfer_write
// CHECK: vector.transfer_write
// ODD: fft-loops-vector-width must be zero or a power of two
func.func @cfft64_forward_q15(%input: tensor<64xi32>) -> tensor<64xi32> {
  %result = ondrix.cfft %input {
    direction = #ondrix.cfft_direction<forward>,
    layout = #ondsp.cx_layout<packed_i16_imag_hi_real_lo>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    product = #ondsp.product<full>,
    product_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 15, rounding = nearest_even, overflow = saturate, saturate_to = i16>,
    output_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 1, rounding = nearest_even, overflow = saturate, saturate_to = i16>
  } : (tensor<64xi32>) -> tensor<64xi32>
  return %result : tensor<64xi32>
}
