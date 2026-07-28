// RUN: ondrix-opt %s --convert-ondrix-to-ondsp --convert-ondsp-fixed-to-scalar --empty-tensor-to-alloc-tensor --one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map allow-return-allocs" --expand-strided-metadata --lower-affine --convert-scf-to-cf --finalize-memref-to-llvm --convert-arith-to-llvm --convert-cf-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts > %t.mlir
// RUN: ondrix-translate %t.mlir --mlir-to-llvmir > %t.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.ll -o %t.o
// RUN: cc %S/Inputs/cfft64_q15_aot.c %t.o -o %t
// RUN: %t

// RUN: ondrix-opt %s --convert-ondrix-to-ondsp="fft-loops" --convert-ondsp-fixed-to-scalar --empty-tensor-to-alloc-tensor --one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map allow-return-allocs" --expand-strided-metadata --lower-affine --convert-scf-to-cf --finalize-memref-to-llvm --convert-arith-to-llvm --convert-cf-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts > %t.loops.mlir
// RUN: ondrix-translate %t.loops.mlir --mlir-to-llvmir > %t.loops.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.loops.ll -o %t.loops.o
// RUN: cc %S/Inputs/cfft64_q15_aot.c %t.loops.o -o %t.loops
// RUN: %t.loops

// The .loops pipeline exercises the opt-in fft-loops lowering (bit-reversal
// and stage loops over in-memory twiddle tables) against the SAME
// independent reference: the loop form must be bit-identical to the
// unrolled recursion.

// First generated-twiddle extent beyond the frozen 4/8/16 tables, executed
// in both directions against an independent recursive reference that embeds
// mpmath-derived twiddles.

func.func @cfft64_forward_q15(%input: tensor<64xi32>) -> tensor<64xi32>
    attributes {llvm.emit_c_interface} {
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

func.func @cfft64_inverse_q15(%input: tensor<64xi32>) -> tensor<64xi32>
    attributes {llvm.emit_c_interface} {
  %result = ondrix.cfft %input {
    direction = #ondrix.cfft_direction<inverse>,
    layout = #ondsp.cx_layout<packed_i16_imag_hi_real_lo>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    product = #ondsp.product<full>,
    product_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 15, rounding = nearest_even, overflow = saturate, saturate_to = i16>,
    output_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 1, rounding = nearest_even, overflow = saturate, saturate_to = i16>
  } : (tensor<64xi32>) -> tensor<64xi32>
  return %result : tensor<64xi32>
}
