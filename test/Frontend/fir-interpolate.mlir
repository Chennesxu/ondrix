// RUN: ondrix-compile %S/Inputs/q15_fir_interpolate.ox | FileCheck %s
// RUN: ondrix-compile %S/Inputs/q15_fir_interpolate_even_taps.ox | FileCheck %s --check-prefix=EVEN
// RUN: ondrix-compile %S/Inputs/f32_fir_interpolate.ox | FileCheck %s --check-prefix=FP \
// RUN:   --implicit-check-not=accumulator --implicit-check-not=ondsp.fixed \
// RUN:   --implicit-check-not=rounding

// CHECK-LABEL: func.func @q15_fir_interpolate(
// CHECK-SAME: tensor<4xi16>
// CHECK-SAME: tensor<3xi16>
// CHECK-SAME: tensor<9xi16>
// CHECK: tensor.empty() : tensor<9xi16>
// CHECK: ondrix.fir_interpolate
// CHECK-SAME: accumulator = !ondsp.acc<storage = i33, frac = 30, signed, update_overflow = wrap>
// CHECK-SAME: factor = 2
// CHECK-SAME: numeric = #ondsp.fixed<signed, storage = i16, frac = 15>

// EVEN-LABEL: func.func @q15_fir_interpolate_even_taps(
// EVEN: ondrix.fir_interpolate
// EVEN-SAME: accumulator = !ondsp.acc<storage = i33, frac = 30, signed, update_overflow = wrap>

// FP-LABEL: func.func @f32_fir_interpolate(
// FP: ondrix.fir_interpolate
// FP-SAME: factor = 2
// FP-SAME: numeric = #ondsp.fp<format = f32, contract = off>
