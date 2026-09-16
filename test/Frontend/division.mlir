// RUN: ondrix-compile %S/Inputs/q15_division.ox | FileCheck %s
// RUN: ondrix-compile %S/Inputs/q15_division.ox | sed 's/q15_division/kernel/' > %t.infix.mlir
// RUN: ondrix-compile %S/Inputs/q15_division_calls.ox | sed 's/q15_division_calls/kernel/' > %t.calls.mlir
// RUN: diff %t.infix.mlir %t.calls.mlir
// RUN: ondrix-compile %S/Inputs/q31_division.ox | FileCheck %s --check-prefix=WIDE
// RUN: not ondrix-compile %S/Inputs/invalid_division_zero.ox 2>&1 | FileCheck %s --check-prefix=ZERO
// RUN: not ondrix-compile %S/Inputs/invalid_division_range.ox 2>&1 | FileCheck %s --check-prefix=RANGE
// RUN: not ondrix-compile %S/Inputs/invalid_division_negative.ox 2>&1 | FileCheck %s --check-prefix=RANGE

// `/` is `div` by a positive integer constant under the language defaults,
// at the precedence of `*` and left-associative with it; the call spelling
// carries a per-site policy, and the two spellings diff clean above.
// CHECK-LABEL: func.func @q15_division(
// CHECK-SAME: %[[X:[^:]+]]: tensor<32xi16>, %[[Y:[^:]+]]: tensor<32xi16>
// CHECK: %[[THIRD:.*]] = ondrix.div %[[X]]
// CHECK-SAME: divisor = 3
// CHECK-SAME: rounding = #ondsp.rounding<nearest_ties_positive>
// CHECK: %[[PRODUCT:.*]] = ondrix.mult %[[Y]], %[[X]]
// CHECK: %[[QUARTER:.*]] = ondrix.div %[[PRODUCT]]
// CHECK-SAME: divisor = 4
// CHECK: %[[SUM:.*]] = ondrix.add %[[THIRD]], %[[QUARTER]]
// CHECK: %[[SIXTH:.*]] = ondrix.div %[[X]]
// CHECK-SAME: divisor = 6
// CHECK-SAME: rounding = #ondsp.rounding<nearest_even>
// CHECK: ondrix.sub %[[SUM]], %[[SIXTH]]

// The divisor range follows the width, and every declared policy reaches the
// operation.
// WIDE-LABEL: func.func @q31_division(
// WIDE: ondrix.div
// WIDE-SAME: divisor = 2147483647
// WIDE-SAME: storage = i32, frac = 31
// WIDE-SAME: overflow = #ondsp.overflow<wrap>
// WIDE-SAME: rounding = #ondsp.rounding<toward_zero>
// WIDE: ondrix.div
// WIDE-SAME: divisor = 1

// ZERO: error: div divisor must be a positive integer in [1, 32767]
// RANGE: error: div divisor must be a positive integer in [1, 32767]
