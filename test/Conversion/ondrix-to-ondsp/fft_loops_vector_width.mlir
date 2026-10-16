// RUN: ondrix-opt %s --convert-ondrix-to-ondsp="fft-loops fft-loops-vector-width=4" | FileCheck %s
// RUN: ondrix-opt %s --convert-ondrix-to-ondsp="fft-loops fft-loops-vector-width=8" | FileCheck %s --check-prefix=WIDE
// RUN: ondrix-opt %s --convert-ondrix-to-ondsp="fft-loops fft-loops-vector-width=2" | FileCheck %s --check-prefix=FALLBACK --implicit-check-not=vector.shuffle
// RUN: ondrix-opt %s --convert-ondrix-to-ondsp="fft-loops fft-loops-vector-width=16" | FileCheck %s --check-prefix=FALLBACK --implicit-check-not=vector.shuffle
// RUN: not ondrix-opt %s --convert-ondrix-to-ondsp="fft-loops fft-loops-vector-width=6" 2>&1 | FileCheck %s --check-prefix=ODD

// The first two stages cross group boundaries inside an eight-value block.
// Both stages retain their own product/output scales, with one block read and
// one block write. Later stages retain contiguous half-length Vector reads.
// CHECK-LABEL: func.func @cfft64_forward_q15
// CHECK: arith.constant dense<32767> : vector<4xi32>
// CHECK: arith.constant dense<[32767, -2147483648, 32767, -2147483648]> : vector<4xi32>
// CHECK: vector.transfer_read {{.*}} : tensor<64xi32>, vector<8xi32>
// CHECK: vector.shuffle {{.*}} [0, 2, 4, 6]
// CHECK: vector.shuffle {{.*}} [1, 3, 5, 7]
// CHECK: ondsp.cx_butterfly {{.*}} : (vector<4xi32>, vector<4xi32>, vector<4xi32>) -> (vector<4xi32>, vector<4xi32>)
// CHECK: vector.shuffle {{.*}} [0, 4, 1, 5, 2, 6, 3, 7]
// CHECK-NOT: vector.transfer_
// CHECK: vector.shuffle {{.*}} [0, 1, 4, 5]
// CHECK: vector.shuffle {{.*}} [2, 3, 6, 7]
// CHECK: ondsp.cx_butterfly {{.*}} : (vector<4xi32>, vector<4xi32>, vector<4xi32>) -> (vector<4xi32>, vector<4xi32>)
// CHECK: vector.shuffle {{.*}} [0, 1, 4, 5, 2, 3, 6, 7]
// CHECK: vector.transfer_write {{.*}} : vector<8xi32>, tensor<64xi32>
// CHECK: scf.for %{{.*}} = %c2 to %c6 step %c1
// CHECK: scf.for %{{.*}} = %c0 to %{{.*}} step %c4
// CHECK: vector.transfer_read {{.*}} : tensor<64xi32>, vector<4xi32>
// CHECK: vector.transfer_read {{.*}} : tensor<64xi32>, vector<4xi32>
// CHECK: vector.transfer_read {{.*}} : tensor<64xi32>, vector<4xi32>
// CHECK: ondsp.cx_butterfly {{.*}} : (vector<4xi32>, vector<4xi32>, vector<4xi32>) -> (vector<4xi32>, vector<4xi32>)
// CHECK: vector.transfer_write
// CHECK: vector.transfer_write
// WIDE-LABEL: func.func @cfft64_forward_q15
// WIDE: vector.transfer_read {{.*}} : tensor<64xi32>, vector<16xi32>
// WIDE-COUNT-3: ondsp.cx_butterfly {{.*}} : (vector<8xi32>, vector<8xi32>, vector<8xi32>) -> (vector<8xi32>, vector<8xi32>)
// WIDE: vector.transfer_write {{.*}} : vector<16xi32>, tensor<64xi32>
// WIDE: scf.for %{{.*}} = %c3 to %c6 step %c1
// FALLBACK-LABEL: func.func @cfft64_forward_q15
// FALLBACK: ondsp.cx_butterfly {{.*}} : (i32, i32, i32) -> (i32, i32)
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
