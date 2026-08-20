// RUN: ondrix-opt %s --convert-ondrix-to-ondsp --convert-ondsp-fixed-to-scalar --empty-tensor-to-alloc-tensor --one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map allow-return-allocs" --expand-strided-metadata --lower-affine --convert-scf-to-cf --finalize-memref-to-llvm --convert-arith-to-llvm --convert-cf-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts > %t.mlir
// RUN: FileCheck %s --input-file=%t.mlir --implicit-check-not=i128
// RUN: ondrix-translate %t.mlir --mlir-to-llvmir > %t.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.ll -o %t.o
// RUN: cc %S/Inputs/rfft_split_q31_aot.c %t.o -o %t -lm
// RUN: %t

// Object-level differential gate for the half-size split contract at both a
// strided (N=8) and a denser (N=16) twiddle extent; no i128 carrier appears.
// CHECK-NOT: ondrix.
// CHECK-NOT: ondsp.

func.func @rfft_split8_q31(%input: tensor<8xi64>) -> tensor<8xi64>
    attributes {llvm.emit_c_interface} {
  %result = ondrix.rfft_split %input {
    layout = #ondsp.cx_layout<packed_i32_imag_hi_real_lo>,
    input_numeric = #ondsp.fixed<signed, storage = i32, frac = 31>,
    output_numeric = #ondsp.fixed<signed, storage = i32, frac = 31>,
    product = #ondsp.product<full>
  } : (tensor<8xi64>) -> tensor<8xi64>
  return %result : tensor<8xi64>
}

func.func @rfft_split16_q31(%input: tensor<16xi64>) -> tensor<16xi64>
    attributes {llvm.emit_c_interface} {
  %result = ondrix.rfft_split %input {
    layout = #ondsp.cx_layout<packed_i32_imag_hi_real_lo>,
    input_numeric = #ondsp.fixed<signed, storage = i32, frac = 31>,
    output_numeric = #ondsp.fixed<signed, storage = i32, frac = 31>,
    product = #ondsp.product<full>
  } : (tensor<16xi64>) -> tensor<16xi64>
  return %result : tensor<16xi64>
}
