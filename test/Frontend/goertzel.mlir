// RUN: ondrix-compile %S/Inputs/f32_goertzel_fma.ox | FileCheck %s
// RUN: not ondrix-compile %S/Inputs/invalid_goertzel_q15.ox 2>&1 | FileCheck %s --check-prefix=FIXED
// RUN: not ondrix-compile %S/Inputs/invalid_goertzel_bin.ox 2>&1 | FileCheck %s --check-prefix=BIN

// The bin and the one contract-indexed multiply-add site are both named at
// the call site; f32 goertzel rounds at no boundary of its own.
// CHECK-LABEL: func.func @f32_goertzel_fma(
// CHECK-SAME: tensor<16xf32>) -> tensor<1xf32>
// CHECK: ondrix.goertzel
// CHECK-SAME: bin = 3 : i64
// CHECK-SAME: numeric = #ondsp.fp<format = f32, contract = fma>
// CHECK-NOT: rounding

// The Q15 energy is tensor<1xi64>, a storage width no source type names.
// FIXED: invalid_goertzel_q15.ox:2:10: error: goertzel currently binds only the f32 profile
// BIN: invalid_goertzel_bin.ox:2:10: error: goertzel bin must lie in [0, N/2]
