// RUN: ondrix-compile %S/Inputs/q15_cic_decimate.ox | FileCheck %s
// RUN: ondrix-compile %S/Inputs/q15_cic_decimate_saturate.ox | FileCheck %s --check-prefix=SAT
// RUN: not ondrix-compile %S/Inputs/invalid_cic_rate.ox 2>&1 | FileCheck %s --check-prefix=RATE
// RUN: not ondrix-compile %S/Inputs/invalid_cic_extent.ox 2>&1 | FileCheck %s --check-prefix=EXTENT
// RUN: not ondrix-compile %S/Inputs/invalid_cic_overflow.ox 2>&1 | FileCheck %s --check-prefix=MODE
// RUN: not ondrix-compile %S/Inputs/invalid_cic_missing_overflow.ox 2>&1 | FileCheck %s --check-prefix=MISSING

// CHECK-LABEL: func.func @q15_cic_decimate(
// CHECK-SAME: %[[INPUT:.*]]: tensor<32xi16>) -> tensor<8xi16>
// CHECK: %[[RESULT:.*]] = ondrix.cic_decimate %[[INPUT]]
// CHECK-SAME: overflow = #ondsp.overflow<wrap>
// CHECK-SAME: rounding = #ondsp.rounding<nearest_ties_positive>
// CHECK: return %[[RESULT]] : tensor<8xi16>

// SAT-LABEL: func.func @q15_cic_decimate_saturate(
// SAT: ondrix.cic_decimate
// SAT-SAME: delay = 2 : i64
// SAT-SAME: overflow = #ondsp.overflow<saturate>
// SAT-SAME: rounding = #ondsp.rounding<nearest_ties_positive>

// RATE: error: cic_decimate requires a power-of-two rate in [2, 4096]
// EXTENT: error: cic_decimate input extent must be the rate times the result extent
// MODE: error: unsupported state overflow mode 'truncate'

// The mode has no default because only one value implements the cascade,
// so omitting it is a parse error rather than a silent choice.
// MISSING: error: expected ',' before state_overflow policy
