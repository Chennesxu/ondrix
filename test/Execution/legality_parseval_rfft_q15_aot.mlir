// RUN: ondrix-opt %s --convert-ondrix-to-ondsp --convert-ondsp-fixed-to-scalar --empty-tensor-to-alloc-tensor --one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map allow-return-allocs" --expand-strided-metadata --lower-affine --convert-scf-to-cf --finalize-memref-to-llvm --convert-arith-to-llvm --convert-cf-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts > %t.mlir
// RUN: ondrix-translate %t.mlir --mlir-to-llvmir > %t.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.ll -o %t.o
// RUN: cc %S/Inputs/legality_parseval_rfft_q15_aot.c %t.o -o %t
// RUN: %t

// Legality counterexample, RFFT stage boundaries ONLY: the companion gate
// legality_parseval_q15_aot measures the full magnitude pipeline, which
// stacks the integer square root, i16 saturation, and re-squaring on top
// of the transform. This gate removes all of that: the compiled function
// returns the packed Hermitian bins themselves and the harness forms the
// spectral energy exactly from the raw real/imaginary components, so the
// ONLY quantization between the two energy readings is the RFFT's own
// per-stage product and output requantization. The rescaled identity
// sum(x^2) == 64 * sum_hermitian(re^2 + im^2) fails bit-wise on every
// full-range witness — the staged Q15 scaling alone already breaks the
// exact isometry, before any magnitude quantization is involved — while
// one benign period-4 witness whose stage values requantize exactly
// satisfies it, pinning the failure as input-dependent.

func.func @parseval_rfft_q15(%input: tensor<64xi16>) -> tensor<33xi32>
    attributes {llvm.emit_c_interface} {
  %bins = ondrix.rfft %input {
    layout = #ondsp.cx_layout<packed_i16_imag_hi_real_lo>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    product = #ondsp.product<full>,
    product_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 15, rounding = nearest_even, overflow = saturate, saturate_to = i16>,
    output_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 1, rounding = nearest_even, overflow = saturate, saturate_to = i16>
  } : (tensor<64xi16>) -> tensor<33xi32>
  return %bins : tensor<33xi32>
}
