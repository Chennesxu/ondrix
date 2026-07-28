// RUN: ondrix-compile %S/Inputs/q15_rms.ox | FileCheck %s
// RUN: not ondrix-compile %S/Inputs/invalid_rms_extent.ox 2>&1 | FileCheck %s --check-prefix=EXTENT
// RUN: not ondrix-compile %S/Inputs/invalid_rms_result.ox 2>&1 | FileCheck %s --check-prefix=SINGLETON

// CHECK-LABEL: func.func @q15_rms(
// CHECK-SAME: %[[INPUT:.*]]: tensor<64xi16>) -> tensor<1xi16>
// CHECK: %[[RESULT:.*]] = ondrix.rms %[[INPUT]]
// CHECK-SAME: numeric = #ondsp.fixed<signed, storage = i16, frac = 15>
// CHECK-SAME: rounding = #ondsp.rounding<nearest_even>
// CHECK: return %[[RESULT]] : tensor<1xi16>

// EXTENT: invalid_rms_extent.ox:2:10: error: rms currently requires a power-of-two input extent in [2, 4096]
// SINGLETON: invalid_rms_result.ox:2:10: error: rms returns a single-element tensor
