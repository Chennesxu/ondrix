// RUN: ondrix-compile %S/Inputs/q15_infix_precedence.ox | FileCheck %s --check-prefix=PRECEDENCE
// RUN: ondrix-compile %S/Inputs/q15_infix_precedence.ox | sed 's/q15_infix_precedence/kernel/' > %t.infix.mlir
// RUN: ondrix-compile %S/Inputs/q15_infix_calls.ox | sed 's/q15_infix_calls/kernel/' > %t.calls.mlir
// RUN: diff %t.infix.mlir %t.calls.mlir
// RUN: ondrix-compile %S/Inputs/q15_infix_grouping.ox | FileCheck %s --check-prefix=GROUPING
// RUN: ondrix-compile %S/Inputs/q31_infix.ox | FileCheck %s --check-prefix=WIDE
// RUN: not ondrix-compile %S/Inputs/invalid_infix_unary_minus.ox 2>&1 | FileCheck %s --check-prefix=MINUS
// RUN: not ondrix-compile %S/Inputs/invalid_infix_division.ox 2>&1 | FileCheck %s --check-prefix=DIVISION
// RUN: not ondrix-compile %S/Inputs/invalid_infix_f32.ox 2>&1 | FileCheck %s --check-prefix=FLOAT
// RUN: not ondrix-compile %S/Inputs/invalid_infix_extent.ox 2>&1 | FileCheck %s --check-prefix=EXTENT
// RUN: not ondrix-compile %S/Inputs/invalid_infix_bare_return.ox 2>&1 | FileCheck %s --check-prefix=BARE
// RUN: not ondrix-compile %S/Inputs/invalid_infix_unclosed.ox 2>&1 | FileCheck %s --check-prefix=UNCLOSED
// RUN: not ondrix-compile %S/Inputs/invalid_infix_buffer.ox 2>&1 | FileCheck %s --check-prefix=BUFFER

// `*` binds before `+` and `-`, which associate to the left; every operator
// is one elementwise operation under the language defaults, and the call
// spelling of the same tree (the diff above) is the identical module.
// PRECEDENCE-LABEL: func.func @q15_infix_precedence(
// PRECEDENCE-SAME: %[[X:[^:]+]]: tensor<32xi16>, %[[Y:[^:]+]]: tensor<32xi16>, %[[Z:[^:]+]]: tensor<32xi16>
// PRECEDENCE: %[[PRODUCT:.*]] = ondrix.mult %[[Y]], %[[Z]]
// PRECEDENCE-SAME: overflow = #ondsp.overflow<saturate>
// PRECEDENCE-SAME: rounding = #ondsp.rounding<nearest_ties_positive>
// PRECEDENCE: %[[SUM:.*]] = ondrix.add %[[X]], %[[PRODUCT]]
// PRECEDENCE-SAME: overflow = #ondsp.overflow<saturate>
// PRECEDENCE: %[[HALF:.*]] = ondrix.shift %[[X]]
// PRECEDENCE: ondrix.sub %[[SUM]], %[[HALF]]

// Parentheses group, a local may hold an infix expression, and a builtin call
// with its own policy stands as an operand without losing it.
// GROUPING-LABEL: func.func @q15_infix_grouping(
// GROUPING-SAME: %[[X:[^:]+]]: tensor<16xi16>, %[[Y:[^:]+]]: tensor<16xi16>, %[[Z:[^:]+]]: tensor<16xi16>
// GROUPING: %[[SUM:.*]] = ondrix.add %[[X]], %[[Y]]
// GROUPING: %[[T:.*]] = ondrix.mult %[[SUM]], %[[Z]]
// GROUPING: %[[FIRST:.*]] = ondrix.sub %[[T]], %[[X]]
// GROUPING: %[[WRAP:.*]] = ondrix.mult %[[Y]], %[[Z]]
// GROUPING-SAME: overflow = #ondsp.overflow<wrap>
// GROUPING: ondrix.sub %[[FIRST]], %[[WRAP]]

// The operand width selects the profile, as it does for the call spelling.
// WIDE-LABEL: func.func @q31_infix(
// WIDE: ondrix.mult
// WIDE-SAME: storage = i32, frac = 31
// WIDE: ondrix.add

// MINUS: error: unary minus is not an operator; spell negate(...)
// DIVISION: error: the divisor is not provably positive
// FLOAT: error: elementwise builtins require q15 or q31 operand elements
// EXTENT: error: binary elementwise builtins require operands of the same element type and extent
// BARE: error: expected a builtin call or an infix expression, not a bare name
// UNCLOSED: error: expected ')' to close the parenthesized expression
// BUFFER: error: composable builtins currently require rank-1 tensor operands
