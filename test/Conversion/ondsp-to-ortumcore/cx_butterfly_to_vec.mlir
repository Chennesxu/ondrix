// RUN: ondrix-opt %s --convert-ondsp-to-ortumcore | FileCheck %s

func.func @cx_butterfly(%a: i32, %b: i32, %tw: i32) -> (i32, i32) {
  %0, %1 = ondsp.cx_butterfly %a, %b, %tw {layout = #ondsp.cx_layout<packed_i16_imag_hi_real_lo>, numeric = #ondsp.fixed<signed, storage = i16, frac = 15>, product = #ondsp.product<full>, scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 15, rounding = trunc, overflow = saturate, saturate_to = i16>} : (i32, i32, i32) -> (i32, i32)
  return %0, %1 : i32, i32
}

// CHECK-LABEL: func.func @cx_butterfly
// CHECK: ortumcore.vec_state_init
// CHECK: ortumcore.vec_set_mode
// CHECK: ortumcore.cx_mul
// CHECK: ortumcore.cx_dual_add
// CHECK: ortumcore.cx_dual_sub
// CHECK-NOT: ondsp.cx_butterfly
