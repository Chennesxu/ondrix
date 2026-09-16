// RUN: ondrix-compile %S/Inputs/q15_ratio.ox | FileCheck %s
// RUN: ondrix-compile %S/Inputs/q15_ratio.ox | sed 's/q15_ratio/kernel/' > %t.infix.mlir
// RUN: ondrix-compile %S/Inputs/q15_ratio_calls.ox | sed 's/q15_ratio_calls/kernel/' > %t.calls.mlir
// RUN: diff %t.infix.mlir %t.calls.mlir
// RUN: ondrix-compile %S/Inputs/q15_ratio_saturate.ox | FileCheck %s --check-prefix=SATURATE
// RUN: ondrix-compile %S/Inputs/q31_ratio.ox | FileCheck %s --check-prefix=WIDE
// RUN: not ondrix-compile %S/Inputs/invalid_infix_division.ox 2>&1 | FileCheck %s --check-prefix=UNPROVEN
// RUN: not ondrix-compile %S/Inputs/invalid_ratio_wrap_proof.ox 2>&1 | FileCheck %s --check-prefix=UNPROVEN
// RUN: not ondrix-compile %S/Inputs/invalid_ratio_square.ox 2>&1 | FileCheck %s --check-prefix=UNPROVEN
// RUN: not ondrix-compile %S/Inputs/invalid_ratio_policy.ox 2>&1 | FileCheck %s --check-prefix=POLICY
// RUN: not ondrix-compile %S/Inputs/invalid_literal_numerator.ox 2>&1 | FileCheck %s --check-prefix=CONSTANT

// `x / y` with a tensor divisor is `ratio`; the infix form leaves the
// non-positive policy unspelled, which the checker admits because `y * y + 1`
// is a square plus a positive constant under saturation, and declares trap.
// CHECK-LABEL: func.func @q15_ratio(
// CHECK: %[[SQUARE:.*]] = ondrix.mult %arg1, %arg1
// CHECK: %[[DIVISOR:.*]] = ondrix.offset %[[SQUARE]] {bias = 1
// CHECK: ondrix.ratio %arg0, %[[DIVISOR]]
// CHECK-SAME: nonpositive = #ondsp.nonpositive_divisor<trap>
// CHECK-SAME: rounding = #ondsp.rounding<nearest_ties_positive>

// A spelled policy needs no proof, and every declared policy reaches the op.
// SATURATE: ondrix.ratio %arg0, %arg1
// SATURATE-SAME: nonpositive = #ondsp.nonpositive_divisor<saturate>
// SATURATE-SAME: rounding = #ondsp.rounding<toward_zero>

// The proof follows the width: a magnitude plus one, and a right shift of a
// magnitude plus a positive constant, are positive at q31 too.
// WIDE-LABEL: func.func @q31_ratio(
// WIDE: ondrix.ratio
// WIDE-SAME: nonpositive = #ondsp.nonpositive_divisor<trap>
// WIDE-SAME: storage = i32, frac = 31
// WIDE: ondrix.ratio
// WIDE-SAME: overflow = #ondsp.overflow<wrap>
// WIDE-SAME: rounding = #ondsp.rounding<toward_negative>

// UNPROVEN: error: the divisor is not provably positive (a magnitude or a square of one operand plus a positive constant, under saturation); spell ratio(x, y, nonpositive=trap) or nonpositive=saturate
// POLICY: error: unsupported nonpositive divisor policy 'zero'
// CONSTANT: error: a constant dividend is not an operation; a tensor divides by a constant or by a tensor
