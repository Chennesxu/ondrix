// RUN: ondrix-opt %s --convert-ondsp-fixed-to-scalar | FileCheck %s

// Two squares of i16 components reach exactly 2^31, one past the signed
// container, so the i33 carrier is what makes the sum exact and the i32
// saturation reachable rather than silent.
func.func @power(%packed: i32) -> i32 {
  %out = ondsp.cx_power %packed {
    layout = #ondsp.cx_layout<packed_i16_imag_hi_real_lo>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    output_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 0,
                                rounding = toward_negative, overflow = saturate,
                                saturate_to = i32>
  } : (i32) -> i32
  return %out : i32
}

// CHECK-LABEL: func.func @power
// CHECK: arith.trunci {{.*}} : i32 to i16
// CHECK: arith.shrui
// CHECK: arith.extsi {{.*}} : i16 to i33
// CHECK: arith.muli {{.*}} : i33
// CHECK: arith.muli {{.*}} : i33
// CHECK: arith.addi {{.*}} : i33
// CHECK: arith.trunci {{.*}} : i33 to i32
