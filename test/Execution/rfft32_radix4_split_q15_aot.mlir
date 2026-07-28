// RUN: ondrix-opt %s --convert-ondrix-to-ondsp --convert-ondsp-fixed-to-scalar --empty-tensor-to-alloc-tensor --one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map" --buffer-deallocation --expand-strided-metadata --lower-affine --convert-scf-to-cf --finalize-memref-to-llvm --convert-arith-to-llvm --convert-cf-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts > %t.mlir
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ondrix-translate %t.mlir --mlir-to-llvmir > %t.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.ll -o %t.o
// RUN: cc %S/Inputs/rfft32_radix4_split_q15_aot.c %t.o -o %t
// RUN: %t

// CHECK-NOT: ondrix.
// CHECK-NOT: ondsp.

func.func @rfft32_radix4_split_q15_value(
    %x0: i16, %x1: i16, %x2: i16, %x3: i16,
    %x4: i16, %x5: i16, %x6: i16, %x7: i16,
    %x8: i16, %x9: i16, %x10: i16, %x11: i16,
    %x12: i16, %x13: i16, %x14: i16, %x15: i16,
    %x16: i16, %x17: i16, %x18: i16, %x19: i16,
    %x20: i16, %x21: i16, %x22: i16, %x23: i16,
    %x24: i16, %x25: i16, %x26: i16, %x27: i16,
    %x28: i16, %x29: i16, %x30: i16, %x31: i16,
    %index: index) -> i32 {
  %input = tensor.from_elements %x0, %x1, %x2, %x3, %x4, %x5, %x6, %x7,
      %x8, %x9, %x10, %x11, %x12, %x13, %x14, %x15, %x16, %x17, %x18, %x19,
      %x20, %x21, %x22, %x23, %x24, %x25, %x26, %x27, %x28, %x29, %x30, %x31
      : tensor<32xi16>
  %result = ondrix.rfft_radix4_split %input {
    layout = #ondsp.cx_layout<packed_i16_imag_hi_real_lo>,
    input_numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    output_numeric = #ondsp.fixed<signed, storage = i16, frac = 10>,
    product = #ondsp.product<full>
  } : (tensor<32xi16>) -> tensor<17xi32>
  %value = tensor.extract %result[%index] : tensor<17xi32>
  return %value : i32
}
