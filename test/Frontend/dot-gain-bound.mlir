// RUN: ondrix-compile %S/Inputs/q15_dot_gain_bound.ox | FileCheck %s --check-prefix=SOURCE
// RUN: ondrix-compile %S/Inputs/q15_dot_gain_bound.ox | ondrix-opt --convert-ondrix-to-ondsp | FileCheck %s --check-prefix=LOWERED
// RUN: ondrix-compile %S/Inputs/q15_dot_gain_bound_policy.ox | FileCheck %s --check-prefix=POLICY
// RUN: ondrix-compile %S/Inputs/q15_dot_gain_bound_constexpr.ox | FileCheck %s --check-prefix=CONSTANT
// RUN: not ondrix-compile %S/Inputs/invalid_gain_bound_constexpr.ox 2>&1 | FileCheck %s --check-prefix=OVER
// RUN: not ondrix-compile %S/Inputs/invalid_gain_bound_zero.ox 2>&1 | FileCheck %s --check-prefix=ZERO
// RUN: not ondrix-compile %S/Inputs/invalid_gain_bound_f32.ox 2>&1 | FileCheck %s --check-prefix=FLOAT

// The declaration keeps the default numeric policy and rides the coefficient
// value into ondsp as a precondition in raw units, 2 << 15.
// SOURCE: ondrix.dot {{.*}}gain_bound = 2 : i64
// SOURCE-SAME: numeric = #ondsp.fixed<signed, storage = i16, frac = 15>
// LOWERED: %[[BOUNDED:.*]] = ondsp.assume_l1_bound %arg1 {bound = 65536 : i64}
// LOWERED: ondsp.reduce_mac %{{.*}}, %arg0, %[[BOUNDED]]

// An explicit policy may follow the declaration.
// POLICY: ondrix.dot {{.*}}gain_bound = 3 : i64
// POLICY-SAME: !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>

// A constant table is checked at compile time: 65535 is below 2 << 15, 65536 is not.
// CONSTANT: gain_bound = 2 : i64
// OVER: error: the constexpr coefficients' absolute sum 65536 is not below gain_bound=2
// ZERO: error: gain_bound must lie in [1, 65536]
// FLOAT: error: gain_bound is a fixed-point coefficient precondition
