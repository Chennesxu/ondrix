// RUN: not ondrix-opt %s --convert-ondrix-to-ondsp 2>&1 | FileCheck %s

func.func @invalid_butterfly_layout(%a: i32, %b: i32, %tw: i32) -> (i32, i32) {
  %0, %1 = ondrix.butterfly %a, %b, %tw {layout = "invalid", numeric = #ondsp.fixed<signed, storage = i16, frac = 15>, product = #ondsp.product<full>, scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 15, rounding = trunc, overflow = saturate, saturate_to = i16>} : (i32, i32, i32) -> (i32, i32)
  return %0, %1 : i32, i32
}

// CHECK: failed to legalize operation 'ondrix.butterfly' that was explicitly marked illegal
