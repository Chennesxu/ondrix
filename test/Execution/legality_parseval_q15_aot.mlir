// RUN: ondrix-opt %s --convert-ondrix-to-ondsp --convert-ondsp-fixed-to-scalar --empty-tensor-to-alloc-tensor --one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map allow-return-allocs" --expand-strided-metadata --lower-affine --convert-scf-to-cf --finalize-memref-to-llvm --convert-arith-to-llvm --convert-cf-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts > %t.mlir
// RUN: ondrix-translate %t.mlir --mlir-to-llvmir > %t.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.ll -o %t.o
// RUN: cc %S/Inputs/legality_parseval_q15_aot.c %t.o -o %t -lm
// RUN: %t

// Legality counterexample: Parseval's identity
// sum(x[n]^2) == sum(|X[k]|^2) / N holds over the reals, so replacing
// time-domain energy with spectral energy (or vice versa) is a valid
// real-arithmetic rewrite. Under the staged-scaling Q15 RFFT and the
// explicit magnitude contract, the nominally rescaled identity
// sum(x^2) == 64 * sum_hermitian(mag^2) fails bit-wise on every witness
// input. This gate measures the FULL quantized pipeline: the per-stage
// product and output requantizations of the RFFT, the integer square
// root rounding, the i16 magnitude saturation, and the squaring of the
// already-rounded magnitudes together replace the exact isometry. The
// companion gate legality_parseval_rfft_q15_aot isolates the RFFT stage
// boundaries alone, without any magnitude quantization. The spectral
// side compiles through the full contract; the exact time-domain energy
// is integer arithmetic with no quantization at all.

func.func @parseval_spectrum_q15(%input: tensor<64xi16>) -> tensor<33xi16>
    attributes {llvm.emit_c_interface} {
  %bins = ondrix.rfft %input {
    layout = #ondsp.cx_layout<packed_i16_imag_hi_real_lo>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    product = #ondsp.product<full>,
    product_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 15, rounding = nearest_even, overflow = saturate, saturate_to = i16>,
    output_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 1, rounding = nearest_even, overflow = saturate, saturate_to = i16>
  } : (tensor<64xi16>) -> tensor<33xi32>
  %magnitudes = ondrix.cx_magnitude %bins {
    layout = #ondsp.cx_layout<packed_i16_imag_hi_real_lo>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<33xi32>) -> tensor<33xi16>
  return %magnitudes : tensor<33xi16>
}
