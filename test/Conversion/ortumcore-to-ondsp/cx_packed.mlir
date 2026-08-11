// RUN: ondrix-opt %s --convert-ortumcore-to-ondsp-emulation | FileCheck %s

// The product emulation extracts signed halves, multiplies exactly in i64,
// and narrows each component through the proven round_shift machinery.
func.func @mul_conj(%v: i32, %w: i32) -> i32 {
  %p = ortumcore.cx_mul_conj %v, %w {shift = 15 : i64,
      rounding = #ortumcore<cx_rounding nearest_ties_positive>,
      overflow = #ortumcore<cx_overflow saturate>,
      layout = #ortumcore<cx_layout imag_hi>} : (i32, i32) -> i32
  return %p : i32
}

// CHECK-LABEL: func.func @mul_conj(
// CHECK-COUNT-4: arith.muli {{.*}} : i64
// CHECK: ondsp.round_shift
// CHECK-SAME: post_shift_right = 15
// CHECK-SAME: rounding = nearest_ties_positive
// CHECK-SAME: overflow = saturate
// CHECK-SAME: saturate_to = i16
// CHECK-SAME: (i64) -> i16
// CHECK: ondsp.round_shift
// CHECK: arith.shli
// CHECK: arith.ori
// CHECK-NOT: ortumcore.

func.func @bfly_cross(%a: i32, %b: i32) -> (i32, i32) {
  %o0, %o1 = ortumcore.cx_bfly %a, %b {shift = 1 : i64,
      rounding = #ortumcore<cx_rounding toward_negative>,
      overflow = #ortumcore<cx_overflow wrap>,
      variant = #ortumcore<cx_bfly_variant cross>} : (i32, i32) -> (i32, i32)
  return %o0, %o1 : i32, i32
}

// CHECK-LABEL: func.func @bfly_cross(
// CHECK-COUNT-2: arith.addi {{.*}} : i32
// CHECK: ondsp.round_shift
// CHECK-SAME: post_shift_right = 1
// CHECK-SAME: rounding = toward_negative
// CHECK-SAME: overflow = wrap
// CHECK-SAME: (i32) -> i16
// CHECK-COUNT-3: ondsp.round_shift
// CHECK-NOT: ortumcore.
