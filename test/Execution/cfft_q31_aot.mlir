// RUN: ondrix-opt %s --convert-ondrix-to-ondsp --convert-ondsp-fixed-to-scalar --empty-tensor-to-alloc-tensor --one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map allow-return-allocs" --expand-strided-metadata --lower-affine --convert-scf-to-cf --finalize-memref-to-llvm --convert-arith-to-llvm --convert-cf-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts > %t.mlir
// RUN: FileCheck %s --input-file=%t.mlir --check-prefix=CARRIER --implicit-check-not=i128 --implicit-check-not=ondrix. --implicit-check-not=ondsp.
// RUN: ondrix-translate %t.mlir --mlir-to-llvmir > %t.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.ll -o %t.o
// RUN: cc %S/Inputs/cfft_q31_aot.c %t.o -o %t
// RUN: %t

// RUN: ondrix-opt %s --convert-ondrix-to-ondsp="fft-loops" --convert-ondsp-fixed-to-scalar --empty-tensor-to-alloc-tensor --one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map allow-return-allocs" --expand-strided-metadata --lower-affine --convert-scf-to-cf --finalize-memref-to-llvm --convert-arith-to-llvm --convert-cf-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts > %t.loops.mlir
// RUN: FileCheck %s --input-file=%t.loops.mlir --check-prefix=CARRIER --implicit-check-not=i128 --implicit-check-not=ondrix. --implicit-check-not=ondsp.
// RUN: ondrix-translate %t.loops.mlir --mlir-to-llvmir > %t.loops.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.loops.ll -o %t.loops.o
// RUN: cc %S/Inputs/cfft_q31_aot.c %t.loops.o -o %t.loops
// RUN: %t.loops

// RUN: ondrix-opt %s --convert-ondrix-to-ondsp --convert-ondsp-cx-butterfly-to-ortumcore --convert-ortumcore-to-ondsp-emulation --convert-ondsp-fixed-to-scalar --empty-tensor-to-alloc-tensor --one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map allow-return-allocs" --expand-strided-metadata --lower-affine --convert-scf-to-cf --finalize-memref-to-llvm --convert-arith-to-llvm --convert-cf-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts > %t.target.mlir
// RUN: ondrix-translate %t.target.mlir --mlir-to-llvmir > %t.target.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.target.ll -o %t.target.o
// RUN: cc %S/Inputs/cfft_q31_aot.c %t.target.o -o %t.target.bin
// RUN: %t.target.bin

// RUN: ondrix-opt %s --convert-ondrix-to-ondsp --convert-ondsp-cx-butterfly-to-ortumcore | FileCheck %s --check-prefix=SELECT

// The .loops pipeline runs the opt-in fft-loops lowering of the same profile
// against the SAME independent reference: the loop form must be bit-identical
// to the unrolled recursion, per element and at every extent here. The .target
// pipeline selects every stage butterfly onto the Q31 scalar target's
// capabilities and executes their emulation against that same reference, so the
// selection is gated by execution rather than by inspection alone.

// Nothing generic survives the selection: each butterfly becomes two
// accumulator webs of raw-high MAC steps and four scaled saturating stage
// operations, and the container halves come apart with plain arithmetic
// because the container is a register pair on the target.
// SELECT: ortumcore.q31_mac_add
// SELECT: ortumcore.q31_mac_sub
// SELECT: ortumcore.acc_out
// SELECT: ortumcore.sat_shift_add
// SELECT: ortumcore.sat_shift_sub
// SELECT-NOT: ondsp.cx_butterfly

// Object-level differential gate for the packed-Q31 CFFT profile: extents 4,
// 8, 16, and the maximum supported 64 in both directions plus the
// forward/inverse composition, checked against an independent reference over
// full-scale rails and xorshift trials.

// The profile is the Q31 scalar target's equation, so no exact carrier for a
// combined cross product exists anywhere in the transform: each term floors to
// i32 inside its own i64 product and the stage combines in i34. The i128
// carrier of the full-product selection must therefore be absent — that
// selection keeps its own operation-level gate in
// test/Execution/cx_butterfly_q31_aot.mlir, where an arbitrary SSA twiddle
// reaches 2^63 and the wide carrier is load bearing.
// CARRIER: i34

func.func @cfft4_forward_q31(%input: tensor<4xi64>) -> tensor<4xi64>
    attributes {llvm.emit_c_interface} {
  %result = ondrix.cfft %input {
    direction = #ondrix.cfft_direction<forward>,
    layout = #ondsp.cx_layout<packed_i32_imag_hi_real_lo>,
    numeric = #ondsp.fixed<signed, storage = i32, frac = 31>,
    product = #ondsp.product<high_raw>,
    product_scale = #ondsp.scale<pre_shift_left = 1, post_shift_right = 0, rounding = toward_negative, overflow = saturate, saturate_to = i32>,
    output_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 1, rounding = toward_negative, overflow = saturate, saturate_to = i32>
  } : (tensor<4xi64>) -> tensor<4xi64>
  return %result : tensor<4xi64>
}

func.func @cfft4_inverse_q31(%input: tensor<4xi64>) -> tensor<4xi64>
    attributes {llvm.emit_c_interface} {
  %result = ondrix.cfft %input {
    direction = #ondrix.cfft_direction<inverse>,
    layout = #ondsp.cx_layout<packed_i32_imag_hi_real_lo>,
    numeric = #ondsp.fixed<signed, storage = i32, frac = 31>,
    product = #ondsp.product<high_raw>,
    product_scale = #ondsp.scale<pre_shift_left = 1, post_shift_right = 0, rounding = toward_negative, overflow = saturate, saturate_to = i32>,
    output_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 1, rounding = toward_negative, overflow = saturate, saturate_to = i32>
  } : (tensor<4xi64>) -> tensor<4xi64>
  return %result : tensor<4xi64>
}

func.func @cfft8_forward_q31(%input: tensor<8xi64>) -> tensor<8xi64>
    attributes {llvm.emit_c_interface} {
  %result = ondrix.cfft %input {
    direction = #ondrix.cfft_direction<forward>,
    layout = #ondsp.cx_layout<packed_i32_imag_hi_real_lo>,
    numeric = #ondsp.fixed<signed, storage = i32, frac = 31>,
    product = #ondsp.product<high_raw>,
    product_scale = #ondsp.scale<pre_shift_left = 1, post_shift_right = 0, rounding = toward_negative, overflow = saturate, saturate_to = i32>,
    output_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 1, rounding = toward_negative, overflow = saturate, saturate_to = i32>
  } : (tensor<8xi64>) -> tensor<8xi64>
  return %result : tensor<8xi64>
}

func.func @cfft8_inverse_q31(%input: tensor<8xi64>) -> tensor<8xi64>
    attributes {llvm.emit_c_interface} {
  %result = ondrix.cfft %input {
    direction = #ondrix.cfft_direction<inverse>,
    layout = #ondsp.cx_layout<packed_i32_imag_hi_real_lo>,
    numeric = #ondsp.fixed<signed, storage = i32, frac = 31>,
    product = #ondsp.product<high_raw>,
    product_scale = #ondsp.scale<pre_shift_left = 1, post_shift_right = 0, rounding = toward_negative, overflow = saturate, saturate_to = i32>,
    output_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 1, rounding = toward_negative, overflow = saturate, saturate_to = i32>
  } : (tensor<8xi64>) -> tensor<8xi64>
  return %result : tensor<8xi64>
}

func.func @cfft16_forward_q31(%input: tensor<16xi64>) -> tensor<16xi64>
    attributes {llvm.emit_c_interface} {
  %result = ondrix.cfft %input {
    direction = #ondrix.cfft_direction<forward>,
    layout = #ondsp.cx_layout<packed_i32_imag_hi_real_lo>,
    numeric = #ondsp.fixed<signed, storage = i32, frac = 31>,
    product = #ondsp.product<high_raw>,
    product_scale = #ondsp.scale<pre_shift_left = 1, post_shift_right = 0, rounding = toward_negative, overflow = saturate, saturate_to = i32>,
    output_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 1, rounding = toward_negative, overflow = saturate, saturate_to = i32>
  } : (tensor<16xi64>) -> tensor<16xi64>
  return %result : tensor<16xi64>
}

func.func @cfft16_inverse_q31(%input: tensor<16xi64>) -> tensor<16xi64>
    attributes {llvm.emit_c_interface} {
  %result = ondrix.cfft %input {
    direction = #ondrix.cfft_direction<inverse>,
    layout = #ondsp.cx_layout<packed_i32_imag_hi_real_lo>,
    numeric = #ondsp.fixed<signed, storage = i32, frac = 31>,
    product = #ondsp.product<high_raw>,
    product_scale = #ondsp.scale<pre_shift_left = 1, post_shift_right = 0, rounding = toward_negative, overflow = saturate, saturate_to = i32>,
    output_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 1, rounding = toward_negative, overflow = saturate, saturate_to = i32>
  } : (tensor<16xi64>) -> tensor<16xi64>
  return %result : tensor<16xi64>
}

func.func @cfft64_forward_q31(%input: tensor<64xi64>) -> tensor<64xi64>
    attributes {llvm.emit_c_interface} {
  %result = ondrix.cfft %input {
    direction = #ondrix.cfft_direction<forward>,
    layout = #ondsp.cx_layout<packed_i32_imag_hi_real_lo>,
    numeric = #ondsp.fixed<signed, storage = i32, frac = 31>,
    product = #ondsp.product<high_raw>,
    product_scale = #ondsp.scale<pre_shift_left = 1, post_shift_right = 0, rounding = toward_negative, overflow = saturate, saturate_to = i32>,
    output_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 1, rounding = toward_negative, overflow = saturate, saturate_to = i32>
  } : (tensor<64xi64>) -> tensor<64xi64>
  return %result : tensor<64xi64>
}

func.func @cfft64_inverse_q31(%input: tensor<64xi64>) -> tensor<64xi64>
    attributes {llvm.emit_c_interface} {
  %result = ondrix.cfft %input {
    direction = #ondrix.cfft_direction<inverse>,
    layout = #ondsp.cx_layout<packed_i32_imag_hi_real_lo>,
    numeric = #ondsp.fixed<signed, storage = i32, frac = 31>,
    product = #ondsp.product<high_raw>,
    product_scale = #ondsp.scale<pre_shift_left = 1, post_shift_right = 0, rounding = toward_negative, overflow = saturate, saturate_to = i32>,
    output_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 1, rounding = toward_negative, overflow = saturate, saturate_to = i32>
  } : (tensor<64xi64>) -> tensor<64xi64>
  return %result : tensor<64xi64>
}

// Every intermediate quantization boundary of both transforms is preserved:
// the composition is two full transforms, not a fused one.
func.func @cfft8_round_trip_q31(%input: tensor<8xi64>) -> tensor<8xi64>
    attributes {llvm.emit_c_interface} {
  %spectrum = ondrix.cfft %input {
    direction = #ondrix.cfft_direction<forward>,
    layout = #ondsp.cx_layout<packed_i32_imag_hi_real_lo>,
    numeric = #ondsp.fixed<signed, storage = i32, frac = 31>,
    product = #ondsp.product<high_raw>,
    product_scale = #ondsp.scale<pre_shift_left = 1, post_shift_right = 0, rounding = toward_negative, overflow = saturate, saturate_to = i32>,
    output_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 1, rounding = toward_negative, overflow = saturate, saturate_to = i32>
  } : (tensor<8xi64>) -> tensor<8xi64>
  %result = ondrix.cfft %spectrum {
    direction = #ondrix.cfft_direction<inverse>,
    layout = #ondsp.cx_layout<packed_i32_imag_hi_real_lo>,
    numeric = #ondsp.fixed<signed, storage = i32, frac = 31>,
    product = #ondsp.product<high_raw>,
    product_scale = #ondsp.scale<pre_shift_left = 1, post_shift_right = 0, rounding = toward_negative, overflow = saturate, saturate_to = i32>,
    output_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 1, rounding = toward_negative, overflow = saturate, saturate_to = i32>
  } : (tensor<8xi64>) -> tensor<8xi64>
  return %result : tensor<8xi64>
}
