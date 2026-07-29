// RUN: ondrix-compile %S/Inputs/q15_sine.ox | FileCheck %s
// RUN: ondrix-compile %S/Inputs/q15_cosine.ox | FileCheck %s --check-prefix=COSINE
// RUN: not ondrix-compile %S/Inputs/invalid_sine_extent.ox 2>&1 | FileCheck %s --check-prefix=EXTENT
// RUN: not ondrix-compile %S/Inputs/invalid_cosine_shape.ox 2>&1 | FileCheck %s --check-prefix=SHAPE

// CHECK-LABEL: func.func @q15_sine(
// CHECK-SAME: tensor<64xi16>) -> tensor<64xi16>
// CHECK: %[[SINE:.*]] = ondrix.sine
// CHECK-SAME: numeric = #ondsp.fixed<signed, storage = i16, frac = 15>
// CHECK-SAME: rounding = #ondsp.rounding<nearest_even>
// CHECK: return %[[SINE]] : tensor<64xi16>

// COSINE-LABEL: func.func @q15_cosine(
// COSINE: ondrix.cosine

// EXTENT: sine currently requires an input extent in [1, 4096]

// SHAPE: cosine result extent must equal the input extent
