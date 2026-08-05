// RUN: ondrix-compile %S/Inputs/q15_window_spectrum.ox | FileCheck %s
// RUN: ondrix-compile %S/Inputs/q15_window_family.ox | FileCheck %s --check-prefix=FAMILY
// RUN: not ondrix-compile %S/Inputs/invalid_window_return.ox 2>&1 | FileCheck %s --check-prefix=SLOT
// RUN: not ondrix-compile %S/Inputs/invalid_window_taps.ox 2>&1 | FileCheck %s --check-prefix=TAPCOUNT
// RUN: not ondrix-compile %S/Inputs/invalid_kaiser_beta.ox 2>&1 | FileCheck %s --check-prefix=BETA

// A window is compile-time design intent in the coefficient slot; the tap
// count names the coefficient extent no result type spells here.
// CHECK-LABEL: func.func @q15_window_spectrum(
// CHECK: %[[W:.*]] = ondrix.window_hamming
// CHECK-SAME: numeric = #ondsp.fixed<signed, storage = i16, frac = 15>
// CHECK-SAME: tensor<9xi16>
// CHECK: ondrix.fir_filter %{{.*}}, %[[W]]
// CHECK-SAME: (tensor<40xi16>, tensor<9xi16>, tensor<32xi16>) -> tensor<32xi16>

// FAMILY-LABEL: func.func @q15_window_family(
// FAMILY: ondrix.window_hann
// FAMILY: ondrix.window_blackman
// FAMILY: ondrix.window_kaiser
// FAMILY-SAME: beta_den = 2 : i64
// FAMILY-SAME: beta_num = 17 : i64

// The slot restriction and the two design bounds the operations declare.
// SLOT: invalid_window_return.ox:2:10: error: hamming is a design expression; it is consumed by fir_filter coefficients
// TAPCOUNT: invalid_window_taps.ox:2:10: error: blackman requires a tap count in [2, 4096]
// BETA: invalid_kaiser_beta.ox:2:10: error: kaiser beta must be a positive rational in (0, 50]
