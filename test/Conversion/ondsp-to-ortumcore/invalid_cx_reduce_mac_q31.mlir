// RUN: not ondrix-opt %s --convert-ondsp-to-ortumcore 2>&1 | FileCheck %s

!acc = !ondsp.acc<storage = i64, frac = 62, signed, update_overflow = saturate>

// The target's packed complex carrier holds 16-bit components, so the Q31
// profile has no target route and must fail closed rather than silently
// selecting the narrower one.
func.func @q31_complex_has_no_target_route(
    %real: !acc, %imag: !acc, %lhs: memref<8xi64>, %rhs: memref<8xi64>) -> (!acc, !acc) {
  // CHECK: unsupported accumulator type '!ondsp.acc<storage = i64, frac = 62, signed, update_overflow = saturate>'
  %re, %im = ondsp.cx_reduce_mac %real, %imag, %lhs, %rhs {
    numeric = #ondsp.fixed<signed, storage = i32, frac = 31>,
    layout = #ondsp.cx_layout<packed_i32_imag_hi_real_lo>
  } : (!acc, !acc, memref<8xi64>, memref<8xi64>) -> (!acc, !acc)
  return %re, %im : !acc, !acc
}
