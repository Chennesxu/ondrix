// RUN: ondrix-opt %s --convert-ondrix-to-ondsp | FileCheck %s

// Two rounding boundaries per element: the ratio division (written out
// because its divisor is a runtime value) and the interpolation, which is
// the one round_shift. Everything after them is exact turn arithmetic.

// CHECK-LABEL: func.func @cx_phase
// The table is a compile-time constant with 129 entries, ending on the
// exact eighth turn (8192 = 0x2000, little-endian trailing bytes) so the
// octant fold meets at the diagonal.
// CHECK: arith.constant dense<"0x00000000{{.*}}00200000"> : tensor<129xi32>
// CHECK: arith.divsi
// CHECK-COUNT-1: ondsp.round_shift {{.*}}post_shift_right = 9, rounding = nearest_even
// CHECK-NOT: ondsp.round_shift
// The quadrant unfold selects among exact turn expressions.
// CHECK: arith.select
func.func @cx_phase(%input: tensor<8xi32>) -> tensor<8xi16> {
  %result = ondrix.cx_phase %input {
    layout = #ondsp.cx_layout<packed_i16_imag_hi_real_lo>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    output_numeric = #ondsp.fixed<unsigned, storage = i16, frac = 16>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<8xi32>) -> tensor<8xi16>
  return %result : tensor<8xi16>
}
