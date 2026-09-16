// RUN: ondrix-compile %S/Inputs/q31_dot.ox | FileCheck %s --check-prefix=DOT
// RUN: ondrix-compile %S/Inputs/q31_fir_constexpr.ox | FileCheck %s --check-prefix=FIR
// RUN: ondrix-compile %S/Inputs/q31_fir_constexpr.ox | ondrix-opt --specialize-ondrix-constant-fir | FileCheck %s --check-prefix=SPECIALIZED
// RUN: ondrix-compile %S/Inputs/q31_dot_raw_high.ox | FileCheck %s --check-prefix=RAW
// RUN: ondrix-compile %S/Inputs/q31_fir_raw_high.ox | FileCheck %s --check-prefix=RAWFIR
// RUN: not ondrix-compile %S/Inputs/invalid_raw_high_q15.ox 2>&1 | FileCheck %s --check-prefix=RAWQ15
// RUN: not ondrix-compile %S/Inputs/invalid_raw_high_width.ox 2>&1 | FileCheck %s --check-prefix=RAWWIDTH
// RUN: not ondrix-compile %S/Inputs/invalid_product_selection.ox 2>&1 | FileCheck %s --check-prefix=SELECTION

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

// product=raw_high is the target's native Q31 selection: each term keeps the
// high half at frac 30 in the shared i40 accumulator, and the readout is the
// identity export into the i64 carrier, one exact doubling, and the declared
// narrowing to Q31.
// RAW-LABEL: func.func @q31_dot_raw_high
// RAW: %[[ACC:.*]] = ondrix.dot
// RAW-SAME: product = #ondsp.product<high_raw>
// RAW-SAME: -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
// RAW: %[[WIDE:.*]] = ondsp.acc_export %[[ACC]]
// RAW-SAME: dst = #ondsp.fixed<signed, storage = i64, frac = 30>
// RAW: %[[DOUBLED:.*]] = arith.shli %[[WIDE]], %c1_i64
// RAW: ondsp.round_shift %[[DOUBLED]]
// RAW-SAME: pre_shift_left = 0, post_shift_right = 0, rounding = nearest_even, overflow = saturate, saturate_to = i32
// RAW-SAME: (i64) -> i32

// RAWFIR-LABEL: func.func @q31_fir_raw_high
// RAWFIR: ondrix.fir
// RAWFIR-SAME: product = #ondsp.product<high_raw>
// RAWFIR-SAME: -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>
// RAWFIR: ondsp.round_shift
// RAWFIR-SAME: overflow = wrap, saturate_to = i32

// RAWQ15: error: product=raw_high is the Q31 dot and fir profile
// RAWWIDTH: error: the executable Q31 raw-high profile requires exact accumulator width 40
// SELECTION: error: unsupported product selection 'low'; use full or raw_high
