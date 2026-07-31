// RUN: ondrix-opt %s --convert-ondrix-to-ondsp --canonicalize --cse \
// RUN:   | FileCheck %s --check-prefix=BASELINE
// RUN: ondrix-opt %s --convert-ondrix-to-ondsp --canonicalize --cse \
// RUN:   --forward-ondrix-insert-extract --canonicalize --cse \
// RUN:   | FileCheck %s --check-prefix=SPECTRUM \
// RUN:     --implicit-check-not="tensor<9xi32>"

// Staged real-spectrum consumer: the RFFT lowering assembles the packed
// spectrum with a chain of constant-index writes and the magnitude lowering
// reads it back with constant-index reads. Upstream folding only inspects the
// nearest write, whose index differs, so without this pass the whole
// intermediate stays materialized.

// BASELINE-LABEL: func.func @rfft16_magnitude_q15(
// BASELINE:         tensor.empty() : tensor<9xi32>
// BASELINE:         tensor.insert %{{.*}} : tensor<9xi32>
// BASELINE:         tensor.extract %{{.*}} : tensor<9xi32>

// After forwarding, the intermediate i32 spectrum has no reads left and dies
// with its writes: the implicit-check-not above rejects every `tensor<9xi32>`
// operation, including the `tensor.empty`. The i16 result tensor is a
// different value and keeps its own writes.
//
// The forwarded chain is pinned on bin 1, whose producing butterfly is
// identified by its unique frozen twiddle constant: the magnitude unpacks the
// real and imaginary halves straight out of that butterfly result, with no
// tensor read in between.
// SPECTRUM-LABEL: func.func @rfft16_magnitude_q15(
// SPECTRUM:         %[[TWIDDLE:.*]] = arith.constant -821791166 : i32
// SPECTRUM:         %[[BIN1:.*]], %{{.*}} = ondsp.cx_butterfly %{{.*}}, %{{.*}}, %[[TWIDDLE]]
// SPECTRUM:         %{{.*}} = arith.trunci %[[BIN1]] : i32 to i16
// SPECTRUM:         %{{.*}} = arith.shrui %[[BIN1]], %{{.*}} : i32
// SPECTRUM:         tensor.insert %{{.*}} : tensor<9xi16>
// SPECTRUM:         return %{{.*}} : tensor<9xi16>

func.func @rfft16_magnitude_q15(%input: tensor<16xi16>) -> tensor<9xi16> {
  %bins = ondrix.rfft %input {
    layout = #ondsp.cx_layout<packed_i16_imag_hi_real_lo>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    product = #ondsp.product<full>,
    product_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 15, rounding = nearest_even, overflow = saturate, saturate_to = i16>,
    output_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 1, rounding = nearest_even, overflow = saturate, saturate_to = i16>
  } : (tensor<16xi16>) -> tensor<9xi32>
  %magnitudes = ondrix.cx_magnitude %bins {
    layout = #ondsp.cx_layout<packed_i16_imag_hi_real_lo>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    rounding = #ondsp.rounding<toward_negative>
  } : (tensor<9xi32>) -> tensor<9xi16>
  return %magnitudes : tensor<9xi16>
}
