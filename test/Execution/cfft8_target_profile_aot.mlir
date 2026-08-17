// RUN: ondrix-opt %s --convert-ondrix-to-ondsp --convert-ondsp-cx-butterfly-to-ortumcore | FileCheck %s --check-prefix=MUL
// RUN: ondrix-opt %s --convert-ondrix-to-ondsp --convert-ondsp-cx-butterfly-to-ortumcore | FileCheck %s --check-prefix=GEN
// RUN: ondrix-opt %s --convert-ondrix-to-ondsp --convert-ondsp-cx-butterfly-to-ortumcore | FileCheck %s --check-prefix=CROSS
// RUN: ondrix-opt %s --convert-ondrix-to-ondsp=fft-loops --convert-ondsp-cx-butterfly-to-ortumcore | FileCheck %s --check-prefix=LOOP
// RUN: ondrix-opt %s --convert-ondrix-to-ondsp=fft-loops --convert-ondsp-cx-butterfly-to-ortumcore | FileCheck %s --check-prefix=LOOPGEN
// RUN: ondrix-opt %s --convert-ondrix-to-ondsp --convert-ondsp-fixed-to-scalar --empty-tensor-to-alloc-tensor --one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map" --buffer-deallocation --expand-strided-metadata --lower-affine --convert-scf-to-cf --finalize-memref-to-llvm --convert-arith-to-llvm --convert-cf-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts > %t.scalar.mlir
// RUN: ondrix-translate %t.scalar.mlir --mlir-to-llvmir > %t.scalar.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.scalar.ll -o %t.scalar.o
// RUN: cc %S/Inputs/cfft8_target_profile_aot.c %t.scalar.o -o %t.scalar.bin
// RUN: %t.scalar.bin
// RUN: ondrix-opt %s --convert-ondrix-to-ondsp --convert-ondsp-cx-butterfly-to-ortumcore --convert-ortumcore-to-ondsp-emulation --convert-ondsp-fixed-to-scalar --empty-tensor-to-alloc-tensor --one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map" --buffer-deallocation --expand-strided-metadata --lower-affine --convert-scf-to-cf --finalize-memref-to-llvm --convert-arith-to-llvm --convert-cf-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts > %t.target.mlir
// RUN: ondrix-translate %t.target.mlir --mlir-to-llvmir > %t.target.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.target.ll -o %t.target.o
// RUN: cc %S/Inputs/cfft8_target_profile_aot.c %t.target.o -o %t.target.bin
// RUN: %t.target.bin
// RUN: ondrix-opt %s --convert-ondrix-to-ondsp=fft-loops --convert-ondsp-fixed-to-scalar --empty-tensor-to-alloc-tensor --one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map allow-return-allocs" --buffer-deallocation --expand-strided-metadata --lower-affine --convert-scf-to-cf --finalize-memref-to-llvm --convert-arith-to-llvm --convert-cf-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts > %t.loop.mlir
// RUN: ondrix-translate %t.loop.mlir --mlir-to-llvmir > %t.loop.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.loop.ll -o %t.loop.o
// RUN: cc %S/Inputs/cfft8_target_profile_aot.c %t.loop.o -o %t.loop.bin
// RUN: %t.loop.bin
// RUN: ondrix-opt %s --convert-ondrix-to-ondsp=fft-loops --convert-ondsp-cx-butterfly-to-ortumcore --convert-ortumcore-to-ondsp-emulation --convert-ondsp-fixed-to-scalar --empty-tensor-to-alloc-tensor --one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map allow-return-allocs" --buffer-deallocation --expand-strided-metadata --lower-affine --convert-scf-to-cf --finalize-memref-to-llvm --convert-arith-to-llvm --convert-cf-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts > %t.looptarget.mlir
// RUN: ondrix-translate %t.looptarget.mlir --mlir-to-llvmir > %t.looptarget.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.looptarget.ll -o %t.looptarget.o
// RUN: cc %S/Inputs/cfft8_target_profile_aot.c %t.looptarget.o -o %t.looptarget.bin
// RUN: %t.looptarget.bin

// Per function: 12 stage butterflies in the paired inventory equation
// (8 plain and 4 cross legs), every one reaching the packed pair - the -j
// twiddle never appears, so nothing stays generic. The loop form carries
// the same equation through its quarter twiddle table: statically one
// plain and one cross butterfly per function body.
// MUL-COUNT-48: ortumcore.cx_mul_conj
// MUL-NOT: ortumcore.cx_mul_conj
// GEN-NOT: ondsp.cx_butterfly
// CROSS-COUNT-16: #ortumcore<cx_bfly_variant cross>
// CROSS-NOT: #ortumcore<cx_bfly_variant cross>
// LOOP-COUNT-8: ortumcore.cx_mul_conj
// LOOP-NOT: ortumcore.cx_mul_conj
// LOOPGEN-NOT: ondsp.cx_butterfly

func.func @cfft8_floor_wrap(%x0: i32, %x1: i32, %x2: i32, %x3: i32,
                            %x4: i32, %x5: i32, %x6: i32, %x7: i32,
                            %index: index) -> i32 {
  %input = tensor.from_elements %x0, %x1, %x2, %x3, %x4, %x5, %x6, %x7
      : tensor<8xi32>
  %result = ondrix.cfft %input {
    direction = #ondrix.cfft_direction<forward>,
    layout = #ondsp.cx_layout<packed_i16_imag_hi_real_lo>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    product = #ondsp.product<full>,
    product_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 15, rounding = toward_negative, overflow = wrap, saturate_to = i16>,
    output_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 1, rounding = toward_negative, overflow = wrap, saturate_to = i16>
  } : (tensor<8xi32>) -> tensor<8xi32>
  %value = tensor.extract %result[%index] : tensor<8xi32>
  return %value : i32
}

func.func @cfft8_ntp_sat(%x0: i32, %x1: i32, %x2: i32, %x3: i32,
                         %x4: i32, %x5: i32, %x6: i32, %x7: i32,
                         %index: index) -> i32 {
  %input = tensor.from_elements %x0, %x1, %x2, %x3, %x4, %x5, %x6, %x7
      : tensor<8xi32>
  %result = ondrix.cfft %input {
    direction = #ondrix.cfft_direction<forward>,
    layout = #ondsp.cx_layout<packed_i16_imag_hi_real_lo>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    product = #ondsp.product<full>,
    product_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 15, rounding = nearest_ties_positive, overflow = saturate, saturate_to = i16>,
    output_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 1, rounding = nearest_ties_positive, overflow = saturate, saturate_to = i16>
  } : (tensor<8xi32>) -> tensor<8xi32>
  %value = tensor.extract %result[%index] : tensor<8xi32>
  return %value : i32
}

func.func @cfft8_floor_sat(%x0: i32, %x1: i32, %x2: i32, %x3: i32,
                           %x4: i32, %x5: i32, %x6: i32, %x7: i32,
                           %index: index) -> i32 {
  %input = tensor.from_elements %x0, %x1, %x2, %x3, %x4, %x5, %x6, %x7
      : tensor<8xi32>
  %result = ondrix.cfft %input {
    direction = #ondrix.cfft_direction<forward>,
    layout = #ondsp.cx_layout<packed_i16_imag_hi_real_lo>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    product = #ondsp.product<full>,
    product_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 15, rounding = toward_negative, overflow = saturate, saturate_to = i16>,
    output_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 1, rounding = toward_negative, overflow = saturate, saturate_to = i16>
  } : (tensor<8xi32>) -> tensor<8xi32>
  %value = tensor.extract %result[%index] : tensor<8xi32>
  return %value : i32
}

func.func @cfft8_ntp_wrap(%x0: i32, %x1: i32, %x2: i32, %x3: i32,
                          %x4: i32, %x5: i32, %x6: i32, %x7: i32,
                          %index: index) -> i32 {
  %input = tensor.from_elements %x0, %x1, %x2, %x3, %x4, %x5, %x6, %x7
      : tensor<8xi32>
  %result = ondrix.cfft %input {
    direction = #ondrix.cfft_direction<forward>,
    layout = #ondsp.cx_layout<packed_i16_imag_hi_real_lo>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    product = #ondsp.product<full>,
    product_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 15, rounding = nearest_ties_positive, overflow = wrap, saturate_to = i16>,
    output_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 1, rounding = nearest_ties_positive, overflow = wrap, saturate_to = i16>
  } : (tensor<8xi32>) -> tensor<8xi32>
  %value = tensor.extract %result[%index] : tensor<8xi32>
  return %value : i32
}
