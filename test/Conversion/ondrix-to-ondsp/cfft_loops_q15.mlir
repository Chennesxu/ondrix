// RUN: ondrix-opt %s --convert-ondrix-to-ondsp="fft-loops" | FileCheck %s
// RUN: ondrix-opt %s --convert-ondrix-to-ondsp="fft-loops" | FileCheck %s --check-prefix=PAIRED

// The opt-in loop-form FFT lowering: one bit-reversal permutation loop over
// a constant index table, then a stage loop with an inner butterfly loop
// over a constant packed twiddle table laid out as table[H + j] = W(2H, j).
// Each butterfly is the same scalar ondsp.cx_butterfly as the unrolled
// recursion with identical requantization attributes, so the two lowerings
// are bit-identical per element; only the code shape changes.

// The size-8 forward tables are pinned exactly: bit reversal [0,4,2,6,1,5,3,7]
// and twiddles [unused, W(2,0)=1, W(4,0)=1, W(4,1)=-j,
// W(8,0)=1, W(8,1)=(23170,-23170), W(8,2)=-j, W(8,3)=(-23170,-23170)],
// packed imag-high/real-low.

// CHECK-LABEL: func.func @cfft8_forward_q15
// CHECK-DAG: arith.constant dense<[0, 32767, 32767, -2147483648, 32767, -1518445950, -2147483648, -1518426754]> : tensor<8xi32>
// CHECK-DAG: arith.constant dense<[0, 4, 2, 6, 1, 5, 3, 7]> : tensor<8xi64>
// CHECK: scf.for
// CHECK: tensor.extract
// CHECK: arith.index_cast
// CHECK: tensor.insert
// CHECK: scf.for
// CHECK: arith.shli
// CHECK: scf.for
// CHECK: ondsp.cx_butterfly
// CHECK-NOT: ondsp.cx_butterfly
// CHECK-NOT: ondrix.cfft
func.func @cfft8_forward_q15(%input: tensor<8xi32>) -> tensor<8xi32> {
  %result = ondrix.cfft %input {
    direction = #ondrix.cfft_direction<forward>,
    layout = #ondsp.cx_layout<packed_i16_imag_hi_real_lo>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    product = #ondsp.product<full>,
    product_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 15, rounding = nearest_even, overflow = saturate, saturate_to = i16>,
    output_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 1, rounding = nearest_even, overflow = saturate, saturate_to = i16>
  } : (tensor<8xi32>) -> tensor<8xi32>
  return %result : tensor<8xi32>
}

// The RFFT loop form packs the real input in a loop, runs the loop CFFT
// core, takes the compact Hermitian slice, and canonicalizes the DC and
// Nyquist bins; the IRFFT loop form seeds DC/Nyquist, mirrors the interior
// bins with saturating conjugation in a loop, and truncates in a loop.

// CHECK-LABEL: func.func @rfft16_q15
// CHECK: scf.for
// CHECK: arith.extui
// CHECK: scf.for
// CHECK: scf.for
// CHECK: ondsp.cx_butterfly
// CHECK-NOT: ondsp.cx_butterfly
// CHECK: tensor.extract_slice
// CHECK-NOT: ondrix.rfft
func.func @rfft16_q15(%input: tensor<16xi16>) -> tensor<9xi32> {
  %result = ondrix.rfft %input {
    layout = #ondsp.cx_layout<packed_i16_imag_hi_real_lo>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    product = #ondsp.product<full>,
    product_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 15, rounding = nearest_even, overflow = saturate, saturate_to = i16>,
    output_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 1, rounding = nearest_even, overflow = saturate, saturate_to = i16>
  } : (tensor<16xi16>) -> tensor<9xi32>
  return %result : tensor<9xi32>
}

// CHECK-LABEL: func.func @irfft16_q15
// CHECK: scf.for
// CHECK: scf.for
// CHECK: scf.for
// CHECK: ondsp.cx_butterfly
// CHECK-NOT: ondsp.cx_butterfly
// CHECK: scf.for
// CHECK: arith.trunci
// CHECK-NOT: ondrix.irfft
func.func @irfft16_q15(%input: tensor<9xi32>) -> tensor<16xi16> {
  %result = ondrix.irfft %input {
    layout = #ondsp.cx_layout<packed_i16_imag_hi_real_lo>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    product = #ondsp.product<full>,
    product_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 15, rounding = nearest_even, overflow = saturate, saturate_to = i16>,
    output_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 1, rounding = nearest_even, overflow = saturate, saturate_to = i16>
  } : (tensor<9xi32>) -> tensor<16xi16>
  return %result : tensor<16xi16>
}

// The paired inventory form materializes no reversal table: the group loop
// carries a reversed-carry cursor (step N/2 = the reversed +1) and reads the
// input through it, one walk step after each of its four loads.

// PAIRED-LABEL: func.func @cfft8_paired_floor
// PAIRED-NOT: tensor<8xi64>
// PAIRED: scf.for
// PAIRED-COUNT-4: ondsp.bitrev_add
// PAIRED-NOT: ondsp.bitrev_add
// PAIRED-NOT: tensor<8xi64>
func.func @cfft8_paired_floor(%input: tensor<8xi32>) -> tensor<8xi32> {
  %result = ondrix.cfft %input {
    direction = #ondrix.cfft_direction<forward>,
    layout = #ondsp.cx_layout<packed_i16_imag_hi_real_lo>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    product = #ondsp.product<full>,
    product_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 15, rounding = toward_negative, overflow = wrap, saturate_to = i16>,
    output_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 1, rounding = toward_negative, overflow = wrap, saturate_to = i16>
  } : (tensor<8xi32>) -> tensor<8xi32>
  return %result : tensor<8xi32>
}
