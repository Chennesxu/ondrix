// RUN: ondrix-opt %s -split-input-file -verify-diagnostics

func.func @scale_rejects_float_destination(%input: i16) -> f32 {
  // expected-error@+1 {{scale saturate_to must be a signless integer type}}
  %0 = ondsp.round_shift %input {scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 1, rounding = toward_negative, overflow = saturate, saturate_to = f32>} : (i16) -> f32
  return %0 : f32
}

// -----

func.func @scale_rejects_index_destination(%input: i16) -> index {
  // expected-error@+1 {{scale saturate_to must be a signless integer type}}
  %0 = ondsp.round_shift %input {scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 1, rounding = toward_negative, overflow = saturate, saturate_to = index>} : (i16) -> index
  return %0 : index
}

// -----

func.func @scale_rejects_unsigned_builtin_destination(%input: i16) -> ui16 {
  // expected-error@+1 {{scale saturate_to must be a signless integer type}}
  %0 = ondsp.round_shift %input {scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 1, rounding = toward_negative, overflow = saturate, saturate_to = ui16>} : (i16) -> ui16
  return %0 : ui16
}
