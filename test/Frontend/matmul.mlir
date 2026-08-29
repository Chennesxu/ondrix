// RUN: ondrix-compile %S/Inputs/q15_matmul.ox | FileCheck %s
// RUN: not ondrix-compile %S/Inputs/invalid_matmul_rank.ox 2>&1 | FileCheck %s --check-prefix=RANK
// RUN: not ondrix-compile %S/Inputs/invalid_matmul_inner.ox 2>&1 | FileCheck %s --check-prefix=INNER
// RUN: not ondrix-compile %S/Inputs/invalid_matmul_extent.ox 2>&1 | FileCheck %s --check-prefix=EXTENT

// CHECK-LABEL: func.func @q15_matmul(
// CHECK-SAME: %[[LHS:.*]]: tensor<4x8xi16>, %[[RHS:.*]]: tensor<8x3xi16>) -> tensor<4x3xi16>
// CHECK: %[[PRODUCT:.*]] = ondrix.matmul %[[LHS]], %[[RHS]]
// CHECK-SAME: numeric = #ondsp.fixed<signed, storage = i16, frac = 15>
// CHECK-SAME: rounding = #ondsp.rounding<nearest_even>
// CHECK-SAME: (tensor<4x8xi16>, tensor<8x3xi16>) -> tensor<4x3xi16>
// CHECK: return %[[PRODUCT]] : tensor<4x3xi16>

// RUN: ondrix-compile %S/Inputs/q15_matmul_floor.ox | FileCheck %s --check-prefix=FLOOR
// RUN: not ondrix-compile %S/Inputs/invalid_matmul_rounding.ox 2>&1 | FileCheck %s --check-prefix=ROUNDING

// The requantization boundary's declared mode routes to the op attribute;
// omission keeps the nearest_even default the plain form pins above.
// FLOOR-LABEL: func.func @q15_matmul_floor
// FLOOR: ondrix.matmul
// FLOOR-SAME: rounding = #ondsp.rounding<toward_negative>
// ROUNDING: invalid_matmul_rounding.ox:2:10: error: matmul rounding must be nearest_even, toward_negative, or nearest_ties_positive

// RANK: invalid_matmul_rank.ox:2:10: error: matmul requires rank-2 tensors
// INNER: invalid_matmul_inner.ox:2:10: error: matmul inner extents must match: lhs columns and rhs rows
// EXTENT: invalid_matmul_extent.ox:2:10: error: matmul currently requires all extents in [1, 64]

// RUN: ondrix-compile %S/Inputs/f32_matmul.ox | FileCheck %s --check-prefix=F32
// RUN: not ondrix-compile %S/Inputs/invalid_f32_matmul_contract.ox 2>&1 | FileCheck %s --check-prefix=CONTRACT

// F32-LABEL: func.func @f32_matmul
// F32: ondrix.matmul
// F32-SAME: numeric = #ondsp.fp<format = f32, contract = fma>
// CONTRACT: error: unsupported floating-point contract 'exact'
