// RUN: ondrix-opt %s --convert-ondrix-to-ondsp --convert-ondsp-fixed-to-scalar --empty-tensor-to-alloc-tensor --one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map" --buffer-deallocation --expand-strided-metadata --lower-affine --convert-scf-to-cf --finalize-memref-to-llvm --convert-arith-to-llvm --convert-cf-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts > %t.mlir
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ondrix-translate %t.mlir --mlir-to-llvmir > %t.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.ll -o %t.o
// RUN: cc %S/Inputs/rfft_q15_aot.c %t.o -o %t
// RUN: %t

// CHECK-NOT: ondrix.
// CHECK-NOT: ondsp.

func.func @rfft8_q15_value(
    %x0: i16, %x1: i16, %x2: i16, %x3: i16,
    %x4: i16, %x5: i16, %x6: i16, %x7: i16,
    %index: index) -> i32 {
  %input = tensor.from_elements %x0, %x1, %x2, %x3, %x4, %x5, %x6, %x7
      : tensor<8xi16>
  %result = ondrix.rfft %input {
    layout = #ondsp.cx_layout<packed_i16_imag_hi_real_lo>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    product = #ondsp.product<full>,
    product_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 15, rounding = nearest_even, overflow = saturate, saturate_to = i16>,
    output_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 1, rounding = nearest_even, overflow = saturate, saturate_to = i16>
  } : (tensor<8xi16>) -> tensor<5xi32>
  %value = tensor.extract %result[%index] : tensor<5xi32>
  return %value : i32
}

func.func @rfft16_q15_value(
    %x0: i16, %x1: i16, %x2: i16, %x3: i16,
    %x4: i16, %x5: i16, %x6: i16, %x7: i16,
    %x8: i16, %x9: i16, %x10: i16, %x11: i16,
    %x12: i16, %x13: i16, %x14: i16, %x15: i16,
    %index: index) -> i32 {
  %input = tensor.from_elements
      %x0, %x1, %x2, %x3, %x4, %x5, %x6, %x7,
      %x8, %x9, %x10, %x11, %x12, %x13, %x14, %x15 : tensor<16xi16>
  %result = ondrix.rfft %input {
    layout = #ondsp.cx_layout<packed_i16_imag_hi_real_lo>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    product = #ondsp.product<full>,
    product_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 15, rounding = nearest_even, overflow = saturate, saturate_to = i16>,
    output_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 1, rounding = nearest_even, overflow = saturate, saturate_to = i16>
  } : (tensor<16xi16>) -> tensor<9xi32>
  %value = tensor.extract %result[%index] : tensor<9xi32>
  return %value : i32
}

func.func @irfft8_q15_value(
    %x0: i32, %x1: i32, %x2: i32, %x3: i32, %x4: i32,
    %index: index) -> i16 {
  %input = tensor.from_elements %x0, %x1, %x2, %x3, %x4 : tensor<5xi32>
  %result = ondrix.irfft %input {
    layout = #ondsp.cx_layout<packed_i16_imag_hi_real_lo>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    product = #ondsp.product<full>,
    product_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 15, rounding = nearest_even, overflow = saturate, saturate_to = i16>,
    output_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 1, rounding = nearest_even, overflow = saturate, saturate_to = i16>
  } : (tensor<5xi32>) -> tensor<8xi16>
  %value = tensor.extract %result[%index] : tensor<8xi16>
  return %value : i16
}

func.func @irfft16_q15_value(
    %x0: i32, %x1: i32, %x2: i32, %x3: i32, %x4: i32,
    %x5: i32, %x6: i32, %x7: i32, %x8: i32,
    %index: index) -> i16 {
  %input = tensor.from_elements
      %x0, %x1, %x2, %x3, %x4, %x5, %x6, %x7, %x8 : tensor<9xi32>
  %result = ondrix.irfft %input {
    layout = #ondsp.cx_layout<packed_i16_imag_hi_real_lo>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    product = #ondsp.product<full>,
    product_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 15, rounding = nearest_even, overflow = saturate, saturate_to = i16>,
    output_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 1, rounding = nearest_even, overflow = saturate, saturate_to = i16>
  } : (tensor<9xi32>) -> tensor<16xi16>
  %value = tensor.extract %result[%index] : tensor<16xi16>
  return %value : i16
}

func.func @rfft_round_trip8_q15_value(
    %x0: i16, %x1: i16, %x2: i16, %x3: i16,
    %x4: i16, %x5: i16, %x6: i16, %x7: i16,
    %index: index) -> i16 {
  %input = tensor.from_elements %x0, %x1, %x2, %x3, %x4, %x5, %x6, %x7
      : tensor<8xi16>
  %spectrum = ondrix.rfft %input {
    layout = #ondsp.cx_layout<packed_i16_imag_hi_real_lo>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    product = #ondsp.product<full>,
    product_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 15, rounding = nearest_even, overflow = saturate, saturate_to = i16>,
    output_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 1, rounding = nearest_even, overflow = saturate, saturate_to = i16>
  } : (tensor<8xi16>) -> tensor<5xi32>
  %output = ondrix.irfft %spectrum {
    layout = #ondsp.cx_layout<packed_i16_imag_hi_real_lo>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    product = #ondsp.product<full>,
    product_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 15, rounding = nearest_even, overflow = saturate, saturate_to = i16>,
    output_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 1, rounding = nearest_even, overflow = saturate, saturate_to = i16>
  } : (tensor<5xi32>) -> tensor<8xi16>
  %value = tensor.extract %output[%index] : tensor<8xi16>
  return %value : i16
}

func.func @rfft_round_trip16_q15_value(
    %x0: i16, %x1: i16, %x2: i16, %x3: i16,
    %x4: i16, %x5: i16, %x6: i16, %x7: i16,
    %x8: i16, %x9: i16, %x10: i16, %x11: i16,
    %x12: i16, %x13: i16, %x14: i16, %x15: i16,
    %index: index) -> i16 {
  %input = tensor.from_elements
      %x0, %x1, %x2, %x3, %x4, %x5, %x6, %x7,
      %x8, %x9, %x10, %x11, %x12, %x13, %x14, %x15 : tensor<16xi16>
  %spectrum = ondrix.rfft %input {
    layout = #ondsp.cx_layout<packed_i16_imag_hi_real_lo>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    product = #ondsp.product<full>,
    product_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 15, rounding = nearest_even, overflow = saturate, saturate_to = i16>,
    output_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 1, rounding = nearest_even, overflow = saturate, saturate_to = i16>
  } : (tensor<16xi16>) -> tensor<9xi32>
  %output = ondrix.irfft %spectrum {
    layout = #ondsp.cx_layout<packed_i16_imag_hi_real_lo>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    product = #ondsp.product<full>,
    product_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 15, rounding = nearest_even, overflow = saturate, saturate_to = i16>,
    output_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 1, rounding = nearest_even, overflow = saturate, saturate_to = i16>
  } : (tensor<9xi32>) -> tensor<16xi16>
  %value = tensor.extract %output[%index] : tensor<16xi16>
  return %value : i16
}
