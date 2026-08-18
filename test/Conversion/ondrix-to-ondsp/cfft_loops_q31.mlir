// RUN: ondrix-opt %s --convert-ondrix-to-ondsp="fft-loops" | FileCheck %s

// The loop form follows the packed profile: the value tensor, the twiddle
// table, and each butterfly carry the i64 container instead of i32, and the
// tables are the same frozen Q31 words the unrolled recursion materializes.
// The paired inventory shape is the 16-bit target's alone, so a Q31 stage
// takes the generic form even though both its boundaries round toward
// negative infinity.

// The size-8 forward tables are pinned exactly: bit reversal [0,4,2,6,1,5,3,7]
// and twiddles [unused, W(2,0)=1, W(4,0)=1, W(4,1)=-j, W(8,0)=1,
// W(8,1)=(1518500250,-1518500250), W(8,2)=-j,
// W(8,3)=(-1518500250,-1518500250)], packed imag-high/real-low.

// CHECK-LABEL: func.func @cfft8_forward_q31
// CHECK-DAG: arith.constant dense<[0, 2147483647, 2147483647, -9223372036854775808, 2147483647, -6521908911199323750, -9223372036854775808, -6521908909941356954]> : tensor<8xi64>
// CHECK-DAG: arith.constant dense<[0, 4, 2, 6, 1, 5, 3, 7]> : tensor<8xi64>
// The permute loop, then one stage loop with one inner butterfly loop.
// CHECK: scf.for
// CHECK: tensor.extract
// CHECK: arith.index_cast
// CHECK: tensor.insert
// CHECK: scf.for
// CHECK: arith.shli
// CHECK: scf.for
// CHECK: ondsp.cx_butterfly {{.*}} : (i64, i64, i64) -> (i64, i64)
// CHECK-NOT: ondsp.cx_butterfly
// CHECK-NOT: ondrix.cfft
// The generic form keeps the reversal-table gather; the cursor walk belongs to
// the paired shape only.
// CHECK-NOT: ondsp.bitrev_add
func.func @cfft8_forward_q31(%input: tensor<8xi64>) -> tensor<8xi64> {
  %result = ondrix.cfft %input {
    direction = #ondrix.cfft_direction<forward>,
    layout = #ondsp.cx_layout<packed_i32_imag_hi_real_lo>,
    numeric = #ondsp.fixed<signed, storage = i32, frac = 31>,
    product = #ondsp.product<high_raw>,
    product_scale = #ondsp.scale<pre_shift_left = 1, post_shift_right = 0, rounding = toward_negative, overflow = saturate, saturate_to = i32>,
    output_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 1, rounding = toward_negative, overflow = saturate, saturate_to = i32>
  } : (tensor<8xi64>) -> tensor<8xi64>
  return %result : tensor<8xi64>
}

// The inverse direction reads its own frozen table rather than negating the
// forward one: W(4,1) = +j saturates to 2147483647 in the high half while its
// forward counterpart is exactly -2147483648.
// CHECK-LABEL: func.func @cfft8_inverse_q31
// CHECK-DAG: arith.constant dense<[0, 2147483647, 2147483647, 9223372032559808512, 2147483647, 6521908914236324250, 9223372032559808512, 6521908915494291046]> : tensor<8xi64>
func.func @cfft8_inverse_q31(%input: tensor<8xi64>) -> tensor<8xi64> {
  %result = ondrix.cfft %input {
    direction = #ondrix.cfft_direction<inverse>,
    layout = #ondsp.cx_layout<packed_i32_imag_hi_real_lo>,
    numeric = #ondsp.fixed<signed, storage = i32, frac = 31>,
    product = #ondsp.product<high_raw>,
    product_scale = #ondsp.scale<pre_shift_left = 1, post_shift_right = 0, rounding = toward_negative, overflow = saturate, saturate_to = i32>,
    output_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 1, rounding = toward_negative, overflow = saturate, saturate_to = i32>
  } : (tensor<8xi64>) -> tensor<8xi64>
  return %result : tensor<8xi64>
}

// The maximum supported Q31 extent, where the deeper frozen table rows are
// reachable at all: the loop form must gather the whole stage-64 quarter turn,
// not just the rows the smaller extents share.
// CHECK-LABEL: func.func @cfft64_forward_q31
// CHECK: arith.constant dense<{{.*}}-904048548761160049{{.*}}> : tensor<64xi64>
// CHECK: ondsp.cx_butterfly {{.*}} : (i64, i64, i64) -> (i64, i64)
// CHECK-NOT: ondsp.cx_butterfly
func.func @cfft64_forward_q31(%input: tensor<64xi64>) -> tensor<64xi64> {
  %result = ondrix.cfft %input {
    direction = #ondrix.cfft_direction<forward>,
    layout = #ondsp.cx_layout<packed_i32_imag_hi_real_lo>,
    numeric = #ondsp.fixed<signed, storage = i32, frac = 31>,
    product = #ondsp.product<high_raw>,
    product_scale = #ondsp.scale<pre_shift_left = 1, post_shift_right = 0, rounding = toward_negative, overflow = saturate, saturate_to = i32>,
    output_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 1, rounding = toward_negative, overflow = saturate, saturate_to = i32>
  } : (tensor<64xi64>) -> tensor<64xi64>
  return %result : tensor<64xi64>
}
