// RUN: ondrix-compile %S/Inputs/q15_dct8.ox | FileCheck %s
// RUN: not ondrix-compile %S/Inputs/invalid_dct_extent.ox 2>&1 | FileCheck %s --check-prefix=EXTENT
// RUN: not ondrix-compile %S/Inputs/invalid_dct_element.ox 2>&1 | FileCheck %s --check-prefix=ELEMENT

// CHECK-LABEL: func.func @q15_dct8(
// CHECK-SAME: %[[INPUT:.*]]: tensor<8xi16>) -> tensor<8xi16>
// CHECK: %[[RESULT:.*]] = ondrix.dct %[[INPUT]]
// CHECK-SAME: input_numeric = #ondsp.fixed<signed, storage = i16, frac = 15>
// CHECK-SAME: output_numeric = #ondsp.fixed<signed, storage = i16, frac = 11>
// CHECK: return %[[RESULT]] : tensor<8xi16>

// EXTENT: invalid_dct_extent.ox:2:10: error: dct currently requires a power-of-two input extent in [4, 64]
// ELEMENT: invalid_dct_element.ox:2:10: error: dct requires a Q15 tensor input and result
