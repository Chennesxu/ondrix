// RUN: ondrix-opt %s --convert-ondrix-to-ondsp --convert-ondsp-fixed-to-scalar --empty-tensor-to-alloc-tensor --one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map allow-return-allocs" --expand-strided-metadata --lower-affine --convert-scf-to-cf --finalize-memref-to-llvm --convert-arith-to-llvm --convert-cf-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts > %t.mlir
// RUN: FileCheck %s --input-file=%t.mlir --implicit-check-not=i128
// RUN: ondrix-translate %t.mlir --mlir-to-llvmir > %t.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.ll -o %t.o
// RUN: cc %S/Inputs/rfft_q31_aot.c %t.o -o %t
// RUN: %t

// RUN: ondrix-opt %s --convert-ondrix-to-ondsp="fft-loops" --convert-ondsp-fixed-to-scalar --empty-tensor-to-alloc-tensor --one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map allow-return-allocs" --expand-strided-metadata --lower-affine --convert-scf-to-cf --finalize-memref-to-llvm --convert-arith-to-llvm --convert-cf-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts > %t.loops.mlir
// RUN: ondrix-translate %t.loops.mlir --mlir-to-llvmir > %t.loops.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.loops.ll -o %t.loops.o
// RUN: cc %S/Inputs/rfft_q31_aot.c %t.loops.o -o %t.loops
// RUN: %t.loops

// RUN: ondrix-opt %s --convert-ondrix-to-ondsp="vectorize-static-cfft" --convert-ondsp-fixed-to-scalar --empty-tensor-to-alloc-tensor --one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map allow-return-allocs" --expand-strided-metadata --lower-affine --convert-scf-to-cf --convert-vector-to-llvm --finalize-memref-to-llvm --convert-arith-to-llvm --convert-cf-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts > %t.vector.mlir
// RUN: ondrix-translate %t.vector.mlir --mlir-to-llvmir > %t.vector.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.vector.ll -o %t.vector.o
// RUN: cc %S/Inputs/rfft_q31_aot.c %t.vector.o -o %t.vector
// RUN: %t.vector

// Object-level differential gate for the packed-Q31 real-spectrum contract:
// forward at 8 and the 64-point table ceiling, the inverse mirror with
// saturating conjugation (an interior bin carries imaginary INT32_MIN in both
// extents), and the staged forward-inverse composition — all three lowering
// shapes against one independent reference. The raw-high profile leaves no
// i128 carrier anywhere.
// CHECK-NOT: ondrix.
// CHECK-NOT: ondsp.

func.func @rfft8_q31(%input: tensor<8xi32>) -> tensor<5xi64>
    attributes {llvm.emit_c_interface} {
  %result = ondrix.rfft %input {
    layout = #ondsp.cx_layout<packed_i32_imag_hi_real_lo>,
    numeric = #ondsp.fixed<signed, storage = i32, frac = 31>,
    product = #ondsp.product<high_raw>,
    product_scale = #ondsp.scale<pre_shift_left = 1, post_shift_right = 0, rounding = toward_negative, overflow = saturate, saturate_to = i32>,
    output_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 1, rounding = toward_negative, overflow = saturate, saturate_to = i32>
  } : (tensor<8xi32>) -> tensor<5xi64>
  return %result : tensor<5xi64>
}

func.func @rfft64_q31(%input: tensor<64xi32>) -> tensor<33xi64>
    attributes {llvm.emit_c_interface} {
  %result = ondrix.rfft %input {
    layout = #ondsp.cx_layout<packed_i32_imag_hi_real_lo>,
    numeric = #ondsp.fixed<signed, storage = i32, frac = 31>,
    product = #ondsp.product<high_raw>,
    product_scale = #ondsp.scale<pre_shift_left = 1, post_shift_right = 0, rounding = toward_negative, overflow = saturate, saturate_to = i32>,
    output_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 1, rounding = toward_negative, overflow = saturate, saturate_to = i32>
  } : (tensor<64xi32>) -> tensor<33xi64>
  return %result : tensor<33xi64>
}

func.func @irfft8_q31(%input: tensor<5xi64>) -> tensor<8xi32>
    attributes {llvm.emit_c_interface} {
  %result = ondrix.irfft %input {
    layout = #ondsp.cx_layout<packed_i32_imag_hi_real_lo>,
    numeric = #ondsp.fixed<signed, storage = i32, frac = 31>,
    product = #ondsp.product<high_raw>,
    product_scale = #ondsp.scale<pre_shift_left = 1, post_shift_right = 0, rounding = toward_negative, overflow = saturate, saturate_to = i32>,
    output_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 1, rounding = toward_negative, overflow = saturate, saturate_to = i32>
  } : (tensor<5xi64>) -> tensor<8xi32>
  return %result : tensor<8xi32>
}

func.func @irfft64_q31(%input: tensor<33xi64>) -> tensor<64xi32>
    attributes {llvm.emit_c_interface} {
  %result = ondrix.irfft %input {
    layout = #ondsp.cx_layout<packed_i32_imag_hi_real_lo>,
    numeric = #ondsp.fixed<signed, storage = i32, frac = 31>,
    product = #ondsp.product<high_raw>,
    product_scale = #ondsp.scale<pre_shift_left = 1, post_shift_right = 0, rounding = toward_negative, overflow = saturate, saturate_to = i32>,
    output_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 1, rounding = toward_negative, overflow = saturate, saturate_to = i32>
  } : (tensor<33xi64>) -> tensor<64xi32>
  return %result : tensor<64xi32>
}

// Both transforms stay observable: the composition is a compact spectrum and
// a full mirror, not a fused identity.
func.func @rfft_round_trip8_q31(%input: tensor<8xi32>) -> tensor<8xi32>
    attributes {llvm.emit_c_interface} {
  %spectrum = ondrix.rfft %input {
    layout = #ondsp.cx_layout<packed_i32_imag_hi_real_lo>,
    numeric = #ondsp.fixed<signed, storage = i32, frac = 31>,
    product = #ondsp.product<high_raw>,
    product_scale = #ondsp.scale<pre_shift_left = 1, post_shift_right = 0, rounding = toward_negative, overflow = saturate, saturate_to = i32>,
    output_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 1, rounding = toward_negative, overflow = saturate, saturate_to = i32>
  } : (tensor<8xi32>) -> tensor<5xi64>
  %result = ondrix.irfft %spectrum {
    layout = #ondsp.cx_layout<packed_i32_imag_hi_real_lo>,
    numeric = #ondsp.fixed<signed, storage = i32, frac = 31>,
    product = #ondsp.product<high_raw>,
    product_scale = #ondsp.scale<pre_shift_left = 1, post_shift_right = 0, rounding = toward_negative, overflow = saturate, saturate_to = i32>,
    output_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 1, rounding = toward_negative, overflow = saturate, saturate_to = i32>
  } : (tensor<5xi64>) -> tensor<8xi32>
  return %result : tensor<8xi32>
}
