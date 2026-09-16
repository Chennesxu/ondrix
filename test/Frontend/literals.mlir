// RUN: ondrix-compile %S/Inputs/q15_literals.ox | FileCheck %s
// RUN: ondrix-compile %S/Inputs/q15_literals.ox | sed 's/q15_literals/kernel/' > %t.infix.mlir
// RUN: ondrix-compile %S/Inputs/q15_literals_calls.ox | sed 's/q15_literals_calls/kernel/' > %t.calls.mlir
// RUN: diff %t.infix.mlir %t.calls.mlir
// RUN: ondrix-compile %S/Inputs/q31_literals.ox | FileCheck %s --check-prefix=WIDE
// RUN: not ondrix-compile %S/Inputs/invalid_literal_pair.ox 2>&1 | FileCheck %s --check-prefix=PAIR
// RUN: not ondrix-compile %S/Inputs/invalid_literal_statement.ox 2>&1 | FileCheck %s --check-prefix=STATEMENT
// RUN: not ondrix-compile %S/Inputs/invalid_literal_minus.ox 2>&1 | FileCheck %s --check-prefix=MINUS
// RUN: not ondrix-compile %S/Inputs/invalid_literal_bias_range.ox 2>&1 | FileCheck %s --check-prefix=BIAS
// RUN: not ondrix-compile %S/Inputs/invalid_literal_gain_range.ox 2>&1 | FileCheck %s --check-prefix=GAIN
// RUN: not ondrix-compile %S/Inputs/invalid_literal_call_operand.ox 2>&1 | FileCheck %s --check-prefix=OPERAND
// RUN: not ondrix-compile %S/Inputs/invalid_literal_numerator.ox 2>&1 | FileCheck %s --check-prefix=NUMERATOR
// RUN: not ondrix-compile %S/Inputs/invalid_literal_f32.ox 2>&1 | FileCheck %s --check-prefix=FLOAT

// A literal beside a tensor is the constant form of the operator, read as a
// raw value in the declared format: `+`/`-` are offset, `*` is gain, and the
// nested gain keeps its own tie rule. The call spelling diffs clean above.
// CHECK-LABEL: func.func @q15_literals(
// CHECK-SAME: %[[X:[^:]+]]: tensor<32xi16>, %[[Y:[^:]+]]: tensor<32xi16>
// CHECK: %[[TRIPLE:.*]] = ondrix.gain %[[Y]]
// CHECK-SAME: gain = 3
// CHECK-SAME: rounding = #ondsp.rounding<nearest_ties_positive>
// CHECK: %[[HALF:.*]] = ondrix.gain %[[X]]
// CHECK-SAME: gain = 16384
// CHECK: %[[SUM:.*]] = ondrix.add %[[TRIPLE]], %[[HALF]]
// CHECK: %[[SHIFTED:.*]] = ondrix.offset %[[SUM]]
// CHECK-SAME: bias = -1024
// CHECK: %[[EVEN:.*]] = ondrix.gain %[[X]]
// CHECK-SAME: gain = 16384
// CHECK-SAME: rounding = #ondsp.rounding<nearest_even>
// CHECK: ondrix.sub %[[SHIFTED]], %[[EVEN]]

// The literal is read at the operand's width.
// WIDE-LABEL: func.func @q31_literals(
// WIDE: ondrix.offset
// WIDE-SAME: bias = 1
// WIDE-SAME: storage = i32, frac = 31
// WIDE: ondrix.gain
// WIDE-SAME: gain = 1073741824

// PAIR: error: an operator needs a tensor operand on at least one side
// STATEMENT: error: expected a builtin call or an infix expression, not a bare name or literal
// MINUS: error: a constant minus a tensor is negate then offset, two boundaries; spell offset(negate(x), bias=c)
// BIAS: error: offset bias must be a raw signed Q1.15 value in [-32768, 32767]
// GAIN: error: gain constant must be a raw signed Q1.15 value in [-32768, 32767]
// OPERAND: error: a literal needs a tensor operand beside it
// NUMERATOR: error: '/' takes a positive integer constant divisor
// FLOAT: error: an integer literal is a raw fixed-point value; scale f32 with gain(x, gain=[n, d], contract=...)
