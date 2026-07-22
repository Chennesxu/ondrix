// RUN: ondrix-compile %S/Inputs/q31_dot.ox | FileCheck %s --check-prefix=DOT
// RUN: ondrix-compile %S/Inputs/q31_fir_constexpr.ox | FileCheck %s --check-prefix=FIR
// RUN: ondrix-compile %S/Inputs/q31_fir_constexpr.ox | ondrix-opt --specialize-ondrix-constant-fir | FileCheck %s --check-prefix=SPECIALIZED

// DOT-LABEL: func.func @q31_dot
// DOT-SAME: memref<?xi32>
// DOT-SAME: memref<?xi32>
// DOT: ondrix.dot
// DOT-SAME: numeric = #ondsp.fixed<signed, storage = i32, frac = 31>
// DOT-SAME: !ondsp.acc<storage = i64, frac = 62, signed, update_overflow = saturate>
// DOT: ondsp.acc_export

// FIR: memref.global "private" constant @__ox_q31_fir_constexpr_coefficients
// FIR-SAME: : memref<4xi32>
// FIR-LABEL: func.func @q31_fir_constexpr(%{{.*}}: memref<4xi32>) -> i32
// FIR: ondrix.fir
// FIR-SAME: numeric = #ondsp.fixed<signed, storage = i32, frac = 31>
// FIR-SAME: !ondsp.acc<storage = i64, frac = 62, signed, update_overflow = wrap>

// SPECIALIZED-LABEL: func.func @q31_fir_constexpr
// SPECIALIZED-NOT: ondrix.fir
// SPECIALIZED: ondsp.acc_add_term
