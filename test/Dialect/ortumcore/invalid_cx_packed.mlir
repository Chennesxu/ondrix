// RUN: not ondrix-opt %s --split-input-file 2>&1 | FileCheck %s

func.func @product_shift_too_wide(%v: i32, %w: i32) -> i32 {
  // CHECK: packed complex product shift must lie in [0, 31]
  %p = ortumcore.cx_mul_conj %v, %w {shift = 32 : i64,
      rounding = #ortumcore<cx_rounding toward_negative>,
      overflow = #ortumcore<cx_overflow wrap>,
      layout = #ortumcore<cx_layout imag_hi>} : (i32, i32) -> i32
  return %p : i32
}

// -----

func.func @butterfly_shift_too_wide(%v: i32, %w: i32) -> i32 {
  // CHECK: packed complex butterfly shift must lie in [0, 1]
  %s0, %s1 = ortumcore.cx_bfly %v, %w {shift = 2 : i64,
      rounding = #ortumcore<cx_rounding toward_negative>,
      overflow = #ortumcore<cx_overflow wrap>,
      variant = #ortumcore<cx_bfly_variant plain>} : (i32, i32) -> (i32, i32)
  return %s0 : i32
}
