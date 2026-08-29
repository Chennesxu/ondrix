// RUN: ondrix-compile %S/Inputs/q15_dct8.ox | FileCheck %s
// RUN: not ondrix-compile %S/Inputs/invalid_dct_extent.ox 2>&1 | FileCheck %s --check-prefix=EXTENT
// RUN: not ondrix-compile %S/Inputs/invalid_dct_element.ox 2>&1 | FileCheck %s --check-prefix=ELEMENT

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
// ELEMENT: invalid_dct_element.ox:2:10: error: dct requires a Q15 or f32 tensor input and a matching result

// RUN: ondrix-compile %S/Inputs/f32_dct.ox | FileCheck %s --check-prefix=F32

// F32-LABEL: func.func @f32_dct
// F32: ondrix.dct
// F32-SAME: input_numeric = #ondsp.fp<format = f32, contract = fma>
