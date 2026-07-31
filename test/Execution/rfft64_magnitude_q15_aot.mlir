// RUN: ondrix-opt %s --convert-ondrix-to-ondsp --convert-ondsp-fixed-to-scalar --empty-tensor-to-alloc-tensor --one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map allow-return-allocs" --expand-strided-metadata --lower-affine --convert-scf-to-cf --finalize-memref-to-llvm --convert-arith-to-llvm --convert-cf-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts > %t.mlir
// RUN: ondrix-translate %t.mlir --mlir-to-llvmir > %t.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.ll -o %t.o
// RUN: cc %S/Inputs/rfft64_magnitude_q15_aot.c %t.o -o %t -lm
// RUN: %t
// RUN: ondrix-opt %s --convert-ondrix-to-ondsp --convert-ondsp-fixed-to-scalar="sqrt-estimate" --empty-tensor-to-alloc-tensor --one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map allow-return-allocs" --expand-strided-metadata --lower-affine --convert-scf-to-cf --finalize-memref-to-llvm --convert-math-to-llvm --convert-arith-to-llvm --convert-cf-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts > %t.estimate.mlir
// RUN: ondrix-translate %t.estimate.mlir --mlir-to-llvmir > %t.estimate.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.estimate.ll -o %t.estimate.o
// RUN: cc %S/Inputs/rfft64_magnitude_q15_aot.c %t.estimate.o -o %t.estimate -lm
// RUN: %t.estimate
// The forwarded execution gate below is self-attesting: this RUN pins that
// the pass actually fired at this extent before the object is built — a
// silently inert pass would leave the packed 33xi32 intermediate alive here.
// RUN: ondrix-opt %s --convert-ondrix-to-ondsp --canonicalize --cse --forward-ondrix-insert-extract --canonicalize --cse | FileCheck %s --check-prefix=FORWARDED --implicit-check-not="tensor<33xi32>"
// FORWARDED-LABEL: func.func @rfft64_magnitude_q15
// RUN: ondrix-opt %s --convert-ondrix-to-ondsp --canonicalize --cse --forward-ondrix-insert-extract --canonicalize --cse --convert-ondsp-fixed-to-scalar --empty-tensor-to-alloc-tensor --one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map allow-return-allocs" --expand-strided-metadata --lower-affine --convert-scf-to-cf --finalize-memref-to-llvm --convert-arith-to-llvm --convert-cf-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts > %t.forwarded.mlir
// RUN: ondrix-translate %t.forwarded.mlir --mlir-to-llvmir > %t.forwarded.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.forwarded.ll -o %t.forwarded.o
// RUN: cc %S/Inputs/rfft64_magnitude_q15_aot.c %t.forwarded.o -o %t.forwarded -lm
// RUN: %t.forwarded

// The second pipeline exercises the opt-in sqrt-estimate lowering against
// the SAME independent reference: both roots must be bit-identical.
//
// The third pipeline forwards the staged spectrum reads onto the RFFT
// scalars, deleting the intermediate packed tensor before bufferization.
// Forwarding replaces a read with the very SSA value that was written, so it
// moves no numeric boundary and the object must satisfy the same
// differential checks bit for bit.
//
// Fixed-point magnitude spectrum: 64 real Q15 samples through the staged
// RFFT contract, then exact sums of squares and the explicit integer square
// root, checked against an independent reference with its own
// correction-looped isqrt.

func.func @rfft64_magnitude_q15(%input: tensor<64xi16>) -> tensor<33xi16>
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
