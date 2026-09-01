// RUN: ondrix-compile %S/Inputs/q15_dct8.ox | FileCheck %s
// RUN: not ondrix-compile %S/Inputs/invalid_dct_extent.ox 2>&1 | FileCheck %s --check-prefix=EXTENT
// RUN: not ondrix-compile %S/Inputs/invalid_dct_product_boundary.ox 2>&1 | FileCheck %s --check-prefix=PRODBOUND

// CHECK-LABEL: func.func @q15_dct8(
// CHECK-SAME: %[[INPUT:.*]]: tensor<8xi16>) -> tensor<8xi16>
// CHECK: %[[RESULT:.*]] = ondrix.dct %[[INPUT]]
// CHECK-SAME: input_numeric = #ondsp.fixed<signed, storage = i16, frac = 15>
// CHECK-SAME: output_numeric = #ondsp.fixed<signed, storage = i16, frac = 11>
// CHECK: return %[[RESULT]] : tensor<8xi16>

// RUN: ondrix-compile %S/Inputs/q15_dct8_floor.ox | FileCheck %s --check-prefix=FLOOR
// RUN: not ondrix-compile %S/Inputs/invalid_dct_rounding.ox 2>&1 | FileCheck %s --check-prefix=ROUNDING

// The export boundary's declared mode routes to the op attribute; omission
// keeps the nearest_even default the plain form pins above.
// FLOOR-LABEL: func.func @q15_dct8_floor
// FLOOR: ondrix.dct
// FLOOR-SAME: rounding = #ondsp.rounding<toward_negative>
// ROUNDING: invalid_dct_rounding.ox:2:10: error: dct rounding must be nearest_even, toward_negative, or nearest_ties_positive

// EXTENT: invalid_dct_extent.ox:2:10: error: dct currently requires a power-of-two input extent in [4, 64]
// The element diagnostic this arm used to pin became unreachable when q31 was
// admitted, so it pins the width-derived boundary instead: Q15 has no product
// boundary and must refuse the declaration that Q31 requires.
// PRODBOUND: invalid_dct_product_boundary.ox:2:10: error: dct at this width and extent has no product boundary to round

// RUN: ondrix-compile %S/Inputs/f32_dct.ox | FileCheck %s --check-prefix=F32

// F32-LABEL: func.func @f32_dct
// F32: ondrix.dct
// F32-SAME: input_numeric = #ondsp.fp<format = f32, contract = fma>

// RUN: ondrix-compile %S/Inputs/q31_dct8.ox | FileCheck %s --check-prefix=Q31
// RUN: ondrix-compile %S/Inputs/q31_dct8_floor.ox | FileCheck %s --check-prefix=Q31FLOOR

// The Q31 reading is frac = 30 - log2(N) = 27, and the row sum of 8 Q31
// products needs the derived per-product boundary the Q15 form refuses.
// Q31-LABEL: func.func @q31_dct8(
// Q31-SAME: %[[INPUT:.*]]: tensor<8xi32>) -> tensor<8xi32>
// Q31: ondrix.dct
// Q31-SAME: input_numeric = #ondsp.fixed<signed, storage = i32, frac = 31>
// Q31-SAME: output_numeric = #ondsp.fixed<signed, storage = i32, frac = 27>
// Q31-SAME: product_rounding = #ondsp.rounding<nearest_even>

// Q31FLOOR: ondrix.dct
// Q31FLOOR-SAME: product_rounding = #ondsp.rounding<toward_negative>
// Q31FLOOR-SAME: rounding = #ondsp.rounding<toward_negative>
