// RUN: ondrix-opt %s --convert-ondsp-cx-butterfly-to-ortumcore | FileCheck %s
// RUN: ondrix-opt %s --convert-ondsp-fixed-to-scalar --convert-scf-to-cf --convert-arith-to-llvm --convert-cf-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts > %t.scalar.mlir
// RUN: ondrix-translate %t.scalar.mlir --mlir-to-llvmir > %t.scalar.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.scalar.ll -o %t.scalar.o
// RUN: ondrix-opt %s --convert-ondsp-cx-butterfly-to-ortumcore --convert-ortumcore-to-ondsp-emulation --convert-ondsp-fixed-to-scalar --convert-scf-to-cf --convert-arith-to-llvm --convert-cf-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts > %t.target.mlir
// RUN: ondrix-translate %t.target.mlir --mlir-to-llvmir > %t.target.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.target.ll -o %t.target.o
// RUN: cc %S/Inputs/ondsp_cx_butterfly_target_aot.c %t.scalar.o -o %t.scalar.bin
// RUN: %t.scalar.bin
// RUN: cc %S/Inputs/ondsp_cx_butterfly_target_aot.c %t.target.o -o %t.target.bin
// RUN: %t.target.bin

// The target leg must actually take the packed path: every butterfly here
// declares an inventory profile with a constant twiddle.
// CHECK-COUNT-3: ortumcore.cx_mul_conj
// CHECK-NOT: ondsp.cx_butterfly

func.func private @pack_pair(%o0: i32, %o1: i32) -> i64 {
  %e0 = arith.extui %o0 : i32 to i64
  %e1 = arith.extui %o1 : i32 to i64
  %c32 = arith.constant 32 : i64
  %hi = arith.shli %e1, %c32 : i64
  %r = arith.ori %hi, %e0 : i64
  return %r : i64
}

func.func @bf_floor_wrap(%a: i32, %b: i32) -> i64 {
  %tw = arith.constant 1518511486 : i32  // 0x5A82A57E
  %0, %1 = ondsp.cx_butterfly %a, %b, %tw {
    layout = #ondsp.cx_layout<packed_i16_imag_hi_real_lo>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    product = #ondsp.product<full>,
    product_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 15, rounding = toward_negative, overflow = wrap, saturate_to = i16>,
    output_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 1, rounding = toward_negative, overflow = wrap, saturate_to = i16>
  } : (i32, i32, i32) -> (i32, i32)
  %r = func.call @pack_pair(%0, %1) : (i32, i32) -> i64
  return %r : i64
}

func.func @bf_ntp_sat(%a: i32, %b: i32) -> i64 {
  %tw = arith.constant 196611 : i32  // 0x00030003: reaches product ties
  %0, %1 = ondsp.cx_butterfly %a, %b, %tw {
    layout = #ondsp.cx_layout<packed_i16_imag_hi_real_lo>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    product = #ondsp.product<full>,
    product_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 15, rounding = nearest_ties_positive, overflow = saturate, saturate_to = i16>,
    output_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 1, rounding = nearest_ties_positive, overflow = saturate, saturate_to = i16>
  } : (i32, i32, i32) -> (i32, i32)
  %r = func.call @pack_pair(%0, %1) : (i32, i32) -> i64
  return %r : i64
}

func.func @bf_ntp_wrap_out0(%a: i32, %b: i32) -> i64 {
  %tw = arith.constant 2147450880 : i32  // 0x7FFF8000: rail twiddle
  %0, %1 = ondsp.cx_butterfly %a, %b, %tw {
    layout = #ondsp.cx_layout<packed_i16_imag_hi_real_lo>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    product = #ondsp.product<full>,
    product_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 15, rounding = nearest_ties_positive, overflow = wrap, saturate_to = i16>,
    output_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 0, rounding = nearest_ties_positive, overflow = wrap, saturate_to = i16>
  } : (i32, i32, i32) -> (i32, i32)
  %r = func.call @pack_pair(%0, %1) : (i32, i32) -> i64
  return %r : i64
}
