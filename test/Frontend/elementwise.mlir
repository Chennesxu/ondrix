// RUN: ondrix-compile %S/Inputs/q15_elementwise_chain.ox | FileCheck %s --check-prefix=CHAIN
// RUN: ondrix-compile %S/Inputs/q15_elementwise_square.ox | FileCheck %s --check-prefix=SQUARE
// RUN: not ondrix-compile %S/Inputs/invalid_elementwise_extent.ox 2>&1 | FileCheck %s --check-prefix=EXTENT
// RUN: not ondrix-compile %S/Inputs/invalid_elementwise_amount.ox 2>&1 | FileCheck %s --check-prefix=AMOUNT
// RUN: not ondrix-compile %S/Inputs/invalid_elementwise_f32.ox 2>&1 | FileCheck %s --check-prefix=FLOAT

// Seven members nest freely and each keeps the boundary its call site named;
// omission takes the language's default, unbiased and non-wrapping.
// CHAIN-LABEL: func.func @q15_elementwise_chain(
// CHAIN: ondrix.mult
// CHAIN-SAME: rounding = #ondsp.rounding<nearest_even>
// CHAIN: ondrix.sub
// CHAIN-SAME: overflow = #ondsp.overflow<saturate>
// CHAIN: ondrix.offset
// CHAIN-SAME: bias = 1024
// CHAIN: ondrix.abs
// CHAIN: ondrix.negate
// CHAIN: ondrix.shift
// CHAIN-SAME: amount = -2
// CHAIN-SAME: rounding = #ondsp.rounding<nearest_ties_positive>
// CHAIN: ondrix.add
// CHAIN-SAME: overflow = #ondsp.overflow<wrap>

// A tensor operand is a value, so squaring reads one parameter twice.
// SQUARE-LABEL: func.func @q15_elementwise_square(
// SQUARE: ondrix.mult %[[X:.*]], %[[X]]

// EXTENT: error: binary elementwise builtins require operands of the same Q15 extent
// AMOUNT: error: shift amount must lie in [-15, 15]

// An elementwise IEEE operation has no boundary to declare, so the family is
// fixed point only rather than carrying an f32 profile with no contract.
// FLOAT: error: elementwise builtins require Q15 operand elements
