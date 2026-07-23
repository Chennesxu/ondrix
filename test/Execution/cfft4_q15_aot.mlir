// RUN: ondrix-opt %s --convert-ondrix-to-ondsp --convert-ondsp-fixed-to-scalar --empty-tensor-to-alloc-tensor --one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map" --buffer-deallocation --expand-strided-metadata --lower-affine --convert-scf-to-cf --finalize-memref-to-llvm --convert-arith-to-llvm --convert-cf-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts > %t.mlir
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ondrix-translate %t.mlir --mlir-to-llvmir > %t.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.ll -o %t.o
// RUN: cc %S/Inputs/cfft4_q15_aot.c %t.o -o %t
// RUN: %t

// CHECK-NOT: ondrix.
// CHECK-NOT: ondsp.

func.func @cfft4_q15_value(
    %x0: i32, %x1: i32, %x2: i32, %x3: i32, %index: index) -> i32 {
  %input = tensor.from_elements %x0, %x1, %x2, %x3 : tensor<4xi32>
  %result = ondrix.cfft %input {
    layout = #ondsp.cx_layout<packed_i16_imag_hi_real_lo>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    product = #ondsp.product<full>,
    product_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 15, rounding = nearest_even, overflow = saturate, saturate_to = i16>,
    output_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 1, rounding = nearest_even, overflow = saturate, saturate_to = i16>
  } : (tensor<4xi32>) -> tensor<4xi32>
  %value = tensor.extract %result[%index] : tensor<4xi32>
  return %value : i32
}
