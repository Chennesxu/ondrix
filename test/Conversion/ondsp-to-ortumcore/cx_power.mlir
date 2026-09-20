// RUN: ondrix-opt %s --convert-ondsp-to-ortumcore | FileCheck %s

// The capability reads its sum back through one saturating right shift, and
// a narrower destination composes with that rather than replacing it.
func.func @power_to_i16(%packed: i32) -> i16 {
  %out = ondsp.cx_power %packed {
    layout = #ondsp.cx_layout<packed_i16_imag_hi_real_lo>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    output_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 15,
                                rounding = nearest_ties_positive, overflow = saturate,
                                saturate_to = i16>
  } : (i32) -> i16
  return %out : i16
}

// CHECK-LABEL: func.func @power_to_i16
// CHECK: ortumcore.cx_power
// CHECK-SAME: rounding = #ortumcore<cx_rounding nearest_ties_positive>
// CHECK-SAME: shift = 15
// CHECK: arith.cmpi slt
// CHECK: arith.trunci {{.*}} : i32 to i16

func.func @power_to_i32(%packed: i32) -> i32 {
  %out = ondsp.cx_power %packed {
    layout = #ondsp.cx_layout<packed_i16_imag_hi_real_lo>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    output_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 0,
                                rounding = toward_negative, overflow = saturate,
                                saturate_to = i32>
  } : (i32) -> i32
  return %out : i32
}

// The 32-bit readout IS the capability, so nothing composes with it.
// CHECK-LABEL: func.func @power_to_i32
// CHECK: ortumcore.cx_power
// CHECK-SAME: shift = 0
// CHECK-NEXT: return
