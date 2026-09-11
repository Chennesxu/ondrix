// RUN: ondrix-opt %s --convert-ondrix-to-ondsp --convert-ondsp-fixed-to-scalar --empty-tensor-to-alloc-tensor --one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map allow-return-allocs" --expand-strided-metadata --lower-affine --convert-scf-to-cf --finalize-memref-to-llvm --convert-arith-to-llvm --convert-cf-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts > %t.unrolled.mlir
// RUN: ondrix-translate %t.unrolled.mlir --mlir-to-llvmir > %t.unrolled.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.unrolled.ll -o %t.unrolled.o
// RUN: cc %S/Inputs/cfft_paired_loops_aot.c %t.unrolled.o -o %t.unrolled.bin
// RUN: %t.unrolled.bin > %t.unrolled.txt
// RUN: ondrix-opt %s --convert-ondrix-to-ondsp=fft-loops --convert-ondsp-fixed-to-scalar --empty-tensor-to-alloc-tensor --one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map allow-return-allocs" --expand-strided-metadata --lower-affine --convert-scf-to-cf --finalize-memref-to-llvm --convert-arith-to-llvm --convert-cf-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts > %t.loops.mlir
// RUN: ondrix-translate %t.loops.mlir --mlir-to-llvmir > %t.loops.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.loops.ll -o %t.loops.o
// RUN: cc %S/Inputs/cfft_paired_loops_aot.c %t.loops.o -o %t.loops.bin
// RUN: %t.loops.bin > %t.loops.txt
// RUN: diff %t.unrolled.txt %t.loops.txt
// RUN: ondrix-opt %s --convert-ondrix-to-ondsp=fft-loops --convert-ondsp-cx-butterfly-to-ortumcore --convert-ortumcore-to-ondsp-emulation --convert-ondsp-fixed-to-scalar --empty-tensor-to-alloc-tensor --one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map allow-return-allocs" --expand-strided-metadata --lower-affine --convert-scf-to-cf --finalize-memref-to-llvm --convert-arith-to-llvm --convert-cf-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts > %t.target.mlir
// RUN: ondrix-translate %t.target.mlir --mlir-to-llvmir > %t.target.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.target.ll -o %t.target.o
// RUN: cc %S/Inputs/cfft_paired_loops_aot.c %t.target.o -o %t.target.bin
// RUN: %t.target.bin > %t.target.txt
// RUN: diff %t.unrolled.txt %t.target.txt

// The paired loop route (radix-4 stage pairs plus one leftover stage at 32
// points, two pairs at 64) must stay bit-identical to the unrolled recursion.

func.func @cfft32_floor_wrap(%input: tensor<32xi32>) -> tensor<32xi32>
    attributes {llvm.emit_c_interface} {
  %result = ondrix.cfft %input {
    direction = #ondrix.cfft_direction<forward>,
    layout = #ondsp.cx_layout<packed_i16_imag_hi_real_lo>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    product = #ondsp.product<full>,
    product_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 15, rounding = toward_negative, overflow = wrap, saturate_to = i16>,
    output_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 1, rounding = toward_negative, overflow = wrap, saturate_to = i16>
  } : (tensor<32xi32>) -> tensor<32xi32>
  return %result : tensor<32xi32>
}

func.func @icfft32_ntp_sat(%input: tensor<32xi32>) -> tensor<32xi32>
    attributes {llvm.emit_c_interface} {
  %result = ondrix.cfft %input {
    direction = #ondrix.cfft_direction<inverse>,
    layout = #ondsp.cx_layout<packed_i16_imag_hi_real_lo>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    product = #ondsp.product<full>,
    product_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 15, rounding = nearest_ties_positive, overflow = saturate, saturate_to = i16>,
    output_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 1, rounding = nearest_ties_positive, overflow = saturate, saturate_to = i16>
  } : (tensor<32xi32>) -> tensor<32xi32>
  return %result : tensor<32xi32>
}

func.func @cfft64_ntp_sat(%input: tensor<64xi32>) -> tensor<64xi32>
    attributes {llvm.emit_c_interface} {
  %result = ondrix.cfft %input {
    direction = #ondrix.cfft_direction<forward>,
    layout = #ondsp.cx_layout<packed_i16_imag_hi_real_lo>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    product = #ondsp.product<full>,
    product_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 15, rounding = nearest_ties_positive, overflow = saturate, saturate_to = i16>,
    output_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 1, rounding = nearest_ties_positive, overflow = saturate, saturate_to = i16>
  } : (tensor<64xi32>) -> tensor<64xi32>
  return %result : tensor<64xi32>
}

func.func @icfft64_floor_wrap(%input: tensor<64xi32>) -> tensor<64xi32>
    attributes {llvm.emit_c_interface} {
  %result = ondrix.cfft %input {
    direction = #ondrix.cfft_direction<inverse>,
    layout = #ondsp.cx_layout<packed_i16_imag_hi_real_lo>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    product = #ondsp.product<full>,
    product_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 15, rounding = toward_negative, overflow = wrap, saturate_to = i16>,
    output_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 1, rounding = toward_negative, overflow = wrap, saturate_to = i16>
  } : (tensor<64xi32>) -> tensor<64xi32>
  return %result : tensor<64xi32>
}
