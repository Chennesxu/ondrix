// RUN: ondrix-opt %s | FileCheck %s

// CHECK-LABEL: func.func @cx_packed_ops
func.func @cx_packed_ops(%v: i32, %w: i32) -> (i32, i32) {
  // CHECK: ortumcore.cx_mul_conj
  // CHECK-SAME: layout = #ortumcore<cx_layout imag_hi>
  // CHECK-SAME: rounding = #ortumcore<cx_rounding nearest_ties_positive>
  // CHECK-SAME: shift = 15
  %p = ortumcore.cx_mul_conj %v, %w {shift = 15 : i64,
      rounding = #ortumcore<cx_rounding nearest_ties_positive>,
      overflow = #ortumcore<cx_overflow saturate>,
      layout = #ortumcore<cx_layout imag_hi>} : (i32, i32) -> i32
  // CHECK: ortumcore.cx_mul_conj
  // CHECK-SAME: layout = #ortumcore<cx_layout real_hi>
  %q = ortumcore.cx_mul_conj %v, %w {shift = 0 : i64,
      rounding = #ortumcore<cx_rounding toward_negative>,
      overflow = #ortumcore<cx_overflow wrap>,
      layout = #ortumcore<cx_layout real_hi>} : (i32, i32) -> i32
  // CHECK: ortumcore.cx_bfly
  // CHECK-SAME: variant = #ortumcore<cx_bfly_variant plain>
  %s0, %s1 = ortumcore.cx_bfly %v, %p {shift = 1 : i64,
      rounding = #ortumcore<cx_rounding toward_negative>,
      overflow = #ortumcore<cx_overflow wrap>,
      variant = #ortumcore<cx_bfly_variant plain>} : (i32, i32) -> (i32, i32)
  // CHECK: ortumcore.cx_bfly
  // CHECK-SAME: variant = #ortumcore<cx_bfly_variant cross>
  %t0, %t1 = ortumcore.cx_bfly %s0, %q {shift = 0 : i64,
      rounding = #ortumcore<cx_rounding nearest_ties_positive>,
      overflow = #ortumcore<cx_overflow saturate>,
      variant = #ortumcore<cx_bfly_variant cross>} : (i32, i32) -> (i32, i32)
  return %t0, %t1 : i32, i32
}
