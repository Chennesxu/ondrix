// RUN: ondrix-compile %S/Inputs/q15_fir_decimate.ox | FileCheck %s
// RUN: ondrix-compile %S/Inputs/f32_fir_decimate.ox | FileCheck %s --check-prefix=FP \
// RUN:   --implicit-check-not=accumulator --implicit-check-not=ondsp.fixed \
// RUN:   --implicit-check-not=rounding

// CHECK-LABEL: func.func @q15_fir_decimate(
// CHECK-SAME: tensor<12xi16>
// CHECK-SAME: tensor<5xi16>
// CHECK-SAME: tensor<4xi16>
// CHECK: tensor.empty() : tensor<4xi16>
// CHECK: ondrix.fir_decimate
// CHECK-SAME: accumulator = !ondsp.acc<storage = i34, frac = 30, signed, update_overflow = wrap>
// CHECK-SAME: factor = 2
// CHECK-SAME: numeric = #ondsp.fixed<signed, storage = i16, frac = 15>

// The f32 binding carries the declared contract and none of the accumulator
// or export attributes the fixed profile needs.
// FP-LABEL: func.func @f32_fir_decimate(
// FP: ondrix.fir_decimate
// FP-SAME: factor = 2
// FP-SAME: numeric = #ondsp.fp<format = f32, contract = fma>

// RUN: ondrix-compile %S/Inputs/q31_fir_decimate.ox | FileCheck %s --check-prefix=Q31

// Automatic accumulation stays Q15-only: no exact accumulator exists for K
// products of 62 bits, so a Q31 site declares its carrier and its per-update
// overflow mode instead of inferring one.
// Q31-LABEL: func.func @q31_fir_decimate(
// Q31: ondrix.fir_decimate
// Q31-SAME: accumulator = !ondsp.acc<storage = i64, frac = 62, signed, update_overflow = saturate>
// Q31-SAME: numeric = #ondsp.fixed<signed, storage = i32, frac = 31>
