// RUN: ondrix-compile %S/Inputs/f32_sos_tdf2.ox | FileCheck %s
// RUN: not ondrix-compile %S/Inputs/invalid_sos_tdf2_fast.ox 2>&1 | FileCheck %s --check-prefix=FAST
// RUN: not ondrix-compile %S/Inputs/invalid_sos_tdf2_layout.ox 2>&1 | FileCheck %s --check-prefix=LAYOUT
// RUN: not ondrix-compile %S/Inputs/invalid_sos_tdf2_q15.ox 2>&1 | FileCheck %s --check-prefix=FIXED

// CHECK-LABEL: func.func @f32_sos_tdf2(
// CHECK-SAME: tensor<?xf32>, %{{.*}}: tensor<2x5xf32>, %{{.*}}: tensor<2xf32>,
// CHECK-SAME: tensor<2x2xf32>) -> (tensor<?xf32>, tensor<2x2xf32>)
// CHECK: %[[OUTPUT:.*]], %[[NEXT:.*]] = ondrix.sos_filter_tdf2
// CHECK-SAME: numeric = #ondsp.fp<format = f32, contract = off>
// CHECK: return %[[OUTPUT]], %[[NEXT]]

// The binding admits exactly what the operation contract admits: the biquad
// event graph has no realization gate for fast, and the profile is f32 only.
// FAST: invalid_sos_tdf2_fast.ox:7:10: error: sos_tdf2 admits only contract=off or contract=fma
// LAYOUT: invalid_sos_tdf2_layout.ox:7:10: error: sos_tdf2 requires coefficients [S,5], scales [S], and state [S,2]
// FIXED: invalid_sos_tdf2_q15.ox:7:10: error: sos_tdf2 requires four f32 tensor parameters and two f32 tensor results
