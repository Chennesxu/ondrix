// RUN: ondrix-compile %S/Inputs/q15_matmul.ox | FileCheck %s
// RUN: ondrix-compile %S/Inputs/q31_matmul.ox | FileCheck %s --check-prefix=Q31
// RUN: ondrix-compile %S/Inputs/q31_matmul_floor.ox | FileCheck %s --check-prefix=Q31FLOOR
// RUN: not ondrix-compile %S/Inputs/invalid_matmul_product_rounding.ox 2>&1 | FileCheck %s --check-prefix=PRODBOUND
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

// The binding supplies the product boundary the inner extent forces and lets a
// call site round the two boundaries independently.
// Q31-LABEL: func.func @q31_matmul(
// Q31: ondrix.matmul
// Q31-SAME: numeric = #ondsp.fixed<signed, storage = i32, frac = 31>
// Q31-SAME: product_rounding = #ondsp.rounding<nearest_even>

// Q31FLOOR-LABEL: func.func @q31_matmul_floor(
// Q31FLOOR: ondrix.matmul
// Q31FLOOR-SAME: product_rounding = #ondsp.rounding<toward_negative>
// Q31FLOOR-SAME: rounding = #ondsp.rounding<nearest_ties_positive>

// PRODBOUND: matmul at this width and inner extent has no product boundary to round
