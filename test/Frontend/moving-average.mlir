// RUN: ondrix-compile %S/Inputs/q15_moving_average.ox | FileCheck %s
// RUN: not ondrix-compile %S/Inputs/invalid_moving_average_window.ox 2>&1 | FileCheck %s --check-prefix=WINDOW
// RUN: not ondrix-compile %S/Inputs/invalid_moving_average_shape.ox 2>&1 | FileCheck %s --check-prefix=SHAPE

// CHECK-LABEL: func.func @q15_moving_average(
// CHECK-SAME: %[[INPUT:.*]]: tensor<40xi16>) -> tensor<33xi16>
// CHECK: %[[RESULT:.*]] = ondrix.moving_average %[[INPUT]]
// CHECK-SAME: numeric = #ondsp.fixed<signed, storage = i16, frac = 15>
// CHECK-SAME: window = 8 : i64
// CHECK: return %[[RESULT]] : tensor<33xi16>

// WINDOW: invalid_moving_average_window.ox:2:10: error: moving_average currently requires a power-of-two window in [2, 64]
// SHAPE: invalid_moving_average_shape.ox:2:10: error: static moving_average result extent is incorrect
