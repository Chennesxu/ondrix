// RUN: ondrix-opt %s --convert-ondrix-to-ondsp="fft-loops" --convert-ondsp-fixed-to-scalar --empty-tensor-to-alloc-tensor --one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map allow-return-allocs" --expand-strided-metadata --lower-affine --convert-scf-to-cf --finalize-memref-to-llvm --convert-arith-to-llvm --convert-cf-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts > %t.mlir
// RUN: ondrix-translate %t.mlir --mlir-to-llvmir > %t.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.ll -o %t.o
// RUN: cc %S/Inputs/rfft1024_q15_aot.c %t.o -o %t
// RUN: %t

// The maximum supported RFFT extent as a CI object gate. The fully
// unrolled lowering makes 1024 points a compile-budget problem (one-off
// local evidence: ~15 minutes and a 2 MB object), which is why only the
// loop-form lowering runs this extent in CI: bit-reversal and stage loops
// over in-memory twiddle tables compile in well under a second to a
// kilobyte-scale object while remaining bit-identical to the unrolled
// recursion per element. Checked against the same independent
// mpmath-derived recursive reference as the smaller extents.

func.func @rfft1024_q15(%input: tensor<1024xi16>) -> tensor<513xi32>
    attributes {llvm.emit_c_interface} {
  %bins = ondrix.rfft %input {
    layout = #ondsp.cx_layout<packed_i16_imag_hi_real_lo>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    product = #ondsp.product<full>,
    product_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 15, rounding = nearest_even, overflow = saturate, saturate_to = i16>,
    output_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 1, rounding = nearest_even, overflow = saturate, saturate_to = i16>
  } : (tensor<1024xi16>) -> tensor<513xi32>
  return %bins : tensor<513xi32>
}
