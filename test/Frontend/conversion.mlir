// RUN: ondrix-compile %S/Inputs/q15_widen.ox | FileCheck %s --check-prefix=WIDEN
// RUN: ondrix-compile %S/Inputs/q31_narrow.ox | FileCheck %s --check-prefix=NARROW
// RUN: ondrix-compile %S/Inputs/q31_narrow_even.ox | FileCheck %s --check-prefix=EVEN
// RUN: ondrix-compile %S/Inputs/q15_conversion_chain.ox | FileCheck %s --check-prefix=CHAIN
// RUN: not ondrix-compile %S/Inputs/invalid_narrow_target.ox 2>&1 | FileCheck %s --check-prefix=NARROW-TARGET
// RUN: not ondrix-compile %S/Inputs/invalid_widen_target.ox 2>&1 | FileCheck %s --check-prefix=WIDEN-TARGET
// RUN: not ondrix-compile %S/Inputs/invalid_widen_policy.ox 2>&1 | FileCheck %s --check-prefix=POLICY
// RUN: not ondrix-compile %S/Inputs/invalid_conversion_f32.ox 2>&1 | FileCheck %s --check-prefix=F32
// RUN: not ondrix-compile %S/Inputs/invalid_conversion_implicit.ox 2>&1 | FileCheck %s --check-prefix=IMPLICIT
// RUN: not ondrix-compile %S/Inputs/invalid_conversion_target_missing.ox 2>&1 | FileCheck %s --check-prefix=MISSING

// A widening is exact and carries no policy; the result type is the target's.
// WIDEN-LABEL: func.func @q15_widen(
// WIDEN-SAME: tensor<32xi16>) -> tensor<32xi32>
// WIDEN: ondrix.quantize
// WIDEN-NOT: rounding
// WIDEN-SAME: src = #ondsp.fixed<signed, storage = i16, frac = 15>
// WIDEN-SAME: (tensor<32xi16>) -> tensor<32xi32>

// NARROW-LABEL: func.func @q31_narrow(
// NARROW: ondrix.quantize
// NARROW-SAME: overflow = #ondsp.overflow<saturate>
// NARROW-SAME: rounding = #ondsp.rounding<nearest_ties_positive>
// NARROW-SAME: (tensor<32xi32>) -> tensor<32xi16>

// EVEN: ondrix.quantize
// EVEN-SAME: overflow = #ondsp.overflow<wrap>
// EVEN-SAME: rounding = #ondsp.rounding<nearest_even>

// The checker carries the converted type through the chain: the widened x
// meets the q31 y in a q31 product, and the sum narrows once at the end.
// CHAIN-LABEL: func.func @q15_conversion_chain(
// CHAIN: %[[WX:.*]] = ondrix.quantize %arg0
// CHAIN: %[[PRODUCT:.*]] = ondrix.mult %[[WX]], %arg1 {numeric = #ondsp.fixed<signed, storage = i32, frac = 31>
// CHAIN: %[[HALF:.*]] = ondrix.div %arg0 {divisor = 2
// CHAIN: %[[WHALF:.*]] = ondrix.quantize %[[HALF]]
// CHAIN: %[[SUM:.*]] = ondrix.add %[[PRODUCT]], %[[WHALF]] {numeric = #ondsp.fixed<signed, storage = i32, frac = 31>
// CHAIN: ondrix.quantize %[[SUM]]
// CHAIN-SAME: rounding = #ondsp.rounding<nearest_even>
// CHAIN-SAME: (tensor<32xi32>) -> tensor<32xi16>

// NARROW-TARGET: error: narrow takes a q31 operand to q15
// WIDEN-TARGET: error: widen takes a q15 operand to q31
// POLICY: error: widen is exact and takes no rounding or overflow policy
// F32: error: narrow converts between q15 and q31; an f32 conversion is a separate contract
// IMPLICIT: error: binary elementwise builtins require operands of the same element type and extent
// MISSING: error: expected ',' before the conversion target
