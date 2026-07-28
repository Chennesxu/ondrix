// RUN: not ondrix-opt %s --split-input-file --convert-ondsp-fixed-to-scalar 2>&1 | FileCheck %s

func.func @round_shift_rejects_pre_shift(%input: i32) -> i32 {
  // CHECK: fixed scalar lowering requires round_shift pre_shift_left = 0
  %0 = ondsp.round_shift %input {scale = #ondsp.scale<pre_shift_left = 1, post_shift_right = 1, rounding = toward_negative, overflow = wrap, saturate_to = i32>} : (i32) -> i32
  return %0 : i32
}

// -----

func.func @round_shift_rejects_full_width_shift(%input: i16) -> i16 {
  // CHECK: fixed scalar lowering requires round_shift post_shift_right narrower than the input storage
  %0 = ondsp.round_shift %input {scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 16, rounding = toward_negative, overflow = wrap, saturate_to = i16>} : (i16) -> i16
  return %0 : i16
}

// -----

func.func @round_shift_rejects_widening_destination(%input: i16) -> i32 {
  // CHECK: fixed scalar lowering does not widen round_shift results
  %0 = ondsp.round_shift %input {scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 1, rounding = toward_negative, overflow = wrap, saturate_to = i32>} : (i16) -> i32
  return %0 : i32
}
