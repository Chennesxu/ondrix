// RUN: ondrix-compile %S/Inputs/q15_sine.ox | FileCheck %s
// RUN: ondrix-compile %S/Inputs/q15_cosine.ox | FileCheck %s --check-prefix=COSINE
// RUN: not ondrix-compile %S/Inputs/invalid_sine_extent.ox 2>&1 | FileCheck %s --check-prefix=EXTENT
// RUN: not ondrix-compile %S/Inputs/invalid_cosine_shape.ox 2>&1 | FileCheck %s --check-prefix=SHAPE
// RUN: ondrix-compile %S/Inputs/q31_sine.ox | FileCheck %s --check-prefix=SINE31
// RUN: ondrix-compile %S/Inputs/q31_cosine.ox | FileCheck %s --check-prefix=COSINE31

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

// The phase carries the same width as the value, so the wider binding is one
// source type on both sides rather than a wider result over a Q15 angle.
// SINE31-LABEL: func.func @q31_sine(
// SINE31-SAME: tensor<64xi32>) -> tensor<64xi32>
// SINE31: ondrix.sine
// SINE31-SAME: numeric = #ondsp.fixed<signed, storage = i32, frac = 31>

// COSINE31-LABEL: func.func @q31_cosine(
// COSINE31: ondrix.cosine
// COSINE31-SAME: storage = i32, frac = 31
