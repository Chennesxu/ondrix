// RUN: ondrix-compile %S/Inputs/f32_cfft16.ox | FileCheck %s
// RUN: ondrix-compile %S/Inputs/f32_cfft_round_trip.ox | FileCheck %s --check-prefix=COMPOSE
// RUN: ondrix-compile %S/Inputs/f32_rfft_round_trip.ox | FileCheck %s --check-prefix=REAL
// RUN: not ondrix-compile %S/Inputs/invalid_f32_cfft_rounding.ox 2>&1 | FileCheck %s --check-prefix=BOUNDARY
// RUN: not ondrix-compile %S/Inputs/invalid_f32_cfft_no_contract.ox 2>&1 | FileCheck %s --check-prefix=CONTRACT
// RUN: not ondrix-compile %S/Inputs/invalid_q15_cfft_contract.ox 2>&1 | FileCheck %s --check-prefix=FIXED

// Sixteen declared points become thirty-two elements: complex_f32 is the one
// type whose extent is not its element count.
// CHECK-LABEL: func.func @f32_cfft16(
// CHECK-SAME: %[[INPUT:.*]]: tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[RESULT:.*]] = ondrix.cfft %[[INPUT]]
// CHECK-SAME: layout = #ondsp.cx_layout<interleaved>
// CHECK-SAME: numeric = #ondsp.fp<format = f32, contract = fma>
// CHECK-NOT: product
// CHECK-NOT: scale
// CHECK: return %[[RESULT]] : tensor<32xf32>

// COMPOSE-LABEL: func.func @f32_cfft_round_trip(
// COMPOSE: ondrix.cfft
// COMPOSE-SAME: cfft_direction<forward>
// COMPOSE: ondrix.cfft
// COMPOSE-SAME: cfft_direction<inverse>

// The real spelling carries N/2+1 bins as two elements each.
// REAL-LABEL: func.func @f32_rfft_round_trip(
// REAL-SAME: %{{.*}}: tensor<32xf32>) -> tensor<32xf32>
// REAL: ondrix.rfft %{{.*}} : (tensor<32xf32>) -> tensor<34xf32>
// REAL: ondrix.irfft %{{.*}} : (tensor<34xf32>) -> tensor<32xf32>

// BOUNDARY: a floating-point transform has no stage boundary to round or saturate
// CONTRACT: a floating-point transform requires contract = off, fma, or fast
// FIXED: a fixed-point transform declares no floating-point contract
