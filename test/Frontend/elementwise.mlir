// RUN: ondrix-compile %S/Inputs/q15_elementwise_chain.ox | FileCheck %s --check-prefix=CHAIN
// RUN: ondrix-compile %S/Inputs/q15_elementwise_square.ox | FileCheck %s --check-prefix=SQUARE
// RUN: not ondrix-compile %S/Inputs/invalid_elementwise_extent.ox 2>&1 | FileCheck %s --check-prefix=EXTENT
// RUN: not ondrix-compile %S/Inputs/invalid_elementwise_amount.ox 2>&1 | FileCheck %s --check-prefix=AMOUNT
// RUN: not ondrix-compile %S/Inputs/invalid_elementwise_f32.ox 2>&1 | FileCheck %s --check-prefix=FLOAT
// RUN: ondrix-compile %S/Inputs/q31_elementwise_chain.ox | FileCheck %s --check-prefix=CHAIN31
// RUN: not ondrix-compile %S/Inputs/invalid_elementwise_mixed_width.ox 2>&1 | FileCheck %s --check-prefix=MIXED
// RUN: not ondrix-compile %S/Inputs/invalid_elementwise_q31_amount.ox 2>&1 | FileCheck %s --check-prefix=AMOUNT31
// RUN: not ondrix-compile %S/Inputs/invalid_elementwise_q31_bias.ox 2>&1 | FileCheck %s --check-prefix=BIAS31

// Seven members nest freely and each keeps the boundary its call site named;
// omission takes the language's export default, add-half and non-wrapping.
// CHAIN-LABEL: func.func @q15_elementwise_chain(
// CHAIN: ondrix.mult
// CHAIN-SAME: rounding = #ondsp.rounding<nearest_ties_positive>
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

// EXTENT: error: binary elementwise builtins require operands of the same element type and extent
// AMOUNT: error: shift amount must lie in [-15, 15]

// An elementwise IEEE operation has no boundary to declare, so the family is
// fixed point only rather than carrying an f32 profile with no contract.
// FLOAT: error: elementwise builtins require q15 or q31 operand elements

// The same seven at the wider width, where the two raw attributes take the
// range the format actually has: amount=-31 and bias=2^30 are legal here and
// would be rejected at Q15.
// CHAIN31-LABEL: func.func @q31_elementwise_chain(
// CHAIN31-SAME: tensor<32xi32>
// CHAIN31: ondrix.mult
// CHAIN31-SAME: storage = i32, frac = 31
// CHAIN31: ondrix.sub
// CHAIN31: ondrix.offset
// CHAIN31-SAME: bias = 1073741824
// CHAIN31: ondrix.abs
// CHAIN31: ondrix.negate
// CHAIN31: ondrix.shift
// CHAIN31-SAME: amount = -31
// CHAIN31: ondrix.add
// CHAIN31-SAME: overflow = #ondsp.overflow<wrap>

// Width is part of the operand agreement, not a per-operand choice: a Q31
// left side does not silently promote a Q15 right side.
// MIXED: error: binary elementwise builtins require operands of the same element type and extent

// Both raw attributes move with the width rather than staying pinned, so
// each has a witness one past its own format's edge.
// AMOUNT31: error: shift amount must lie in [-31, 31]
// BIAS31: error: offset bias must be a raw signed Q1.31 value in [-2147483648, 2147483647]
