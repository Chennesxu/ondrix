// RUN: ondrix-opt %s --convert-ondsp-cx-butterfly-to-ortumcore | FileCheck %s

// The capability profile selects; every scale outside it stays generic.

// CHECK-LABEL: func.func @capability_pair
// CHECK: ortumcore.sat_shift_sub %{{.*}} {shift = 1 : i64}
// CHECK: ortumcore.sat_shift_add %{{.*}} {shift = 3 : i64}
// CHECK-NOT: ondsp.sub_shift
// CHECK-NOT: ondsp.add_shift
func.func @capability_pair(%a: i32, %b: i32) -> (i32, i32) {
  %d = ondsp.sub_shift %a, %b {
    scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 1, rounding = toward_negative, overflow = saturate, saturate_to = i32>
  } : (i32, i32) -> i32
  %s = ondsp.add_shift %a, %b {
    scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 3, rounding = toward_negative, overflow = saturate, saturate_to = i32>
  } : (i32, i32) -> i32
  return %d, %s : i32, i32
}

// CHECK-LABEL: func.func @stays_generic
// CHECK-NOT: ortumcore.
// CHECK: ondsp.sub_shift
// CHECK: ondsp.sub_shift
// CHECK: ondsp.sub_shift
// CHECK: ondsp.add_shift
// CHECK: ondsp.sub_shift
func.func @stays_generic(%a: i32, %b: i32, %w: i64) -> (i32, i32, i32, i32, i64) {
  // Toward-zero rounding is not the capability's.
  %r0 = ondsp.sub_shift %a, %b {
    scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 1, rounding = toward_zero, overflow = saturate, saturate_to = i32>
  } : (i32, i32) -> i32
  // Wrapping narrowing is not the capability's.
  %r1 = ondsp.sub_shift %a, %b {
    scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 1, rounding = toward_negative, overflow = wrap, saturate_to = i32>
  } : (i32, i32) -> i32
  // The shift is outside the capability domain.
  %r2 = ondsp.sub_shift %a, %b {
    scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 4, rounding = toward_negative, overflow = saturate, saturate_to = i32>
  } : (i32, i32) -> i32
  // A pre-shift is not the capability's.
  %r3 = ondsp.add_shift %a, %b {
    scale = #ondsp.scale<pre_shift_left = 1, post_shift_right = 1, rounding = toward_negative, overflow = saturate, saturate_to = i32>
  } : (i32, i32) -> i32
  // A 64-bit carrier is not the capability's.
  %r4 = ondsp.sub_shift %w, %w {
    scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 1, rounding = toward_negative, overflow = saturate, saturate_to = i64>
  } : (i64, i64) -> i64
  return %r0, %r1, %r2, %r3, %r4 : i32, i32, i32, i32, i64
}
