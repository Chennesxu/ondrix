// RUN: ondrix-compile %S/Inputs/q15_fir_stream.ox | FileCheck %s
// RUN: ondrix-compile %S/Inputs/q31_fir_stream.ox | FileCheck %s --check-prefix=Q31
// RUN: not ondrix-compile %S/Inputs/invalid_q31_fir_stream_default.ox 2>&1 | FileCheck %s --check-prefix=Q31-DEFAULT

// CHECK-LABEL: func.func @q15_fir_stream(
// CHECK-SAME: tensor<?xi16>
// CHECK-SAME: tensor<3xi16>
// CHECK-SAME: tensor<2xi16>
// CHECK-SAME: -> (tensor<?xi16>, tensor<2xi16>)
// CHECK: %[[OUTPUT:.*]], %[[NEXT:.*]] = ondrix.fir_stream
// CHECK-SAME: accumulator = !ondsp.acc<storage = i33, frac = 30, signed, update_overflow = wrap>
// CHECK-SAME: overflow = #ondsp.overflow<saturate>
// CHECK-SAME: rounding = #ondsp.rounding<nearest_ties_positive>
// CHECK: return %[[OUTPUT]], %[[NEXT]]

// The spelled contract at the width where two products already leave i64.
// Q31-LABEL: func.func @q31_fir_stream(
// Q31-SAME: tensor<?xi32>
// Q31: ondrix.fir_stream
// Q31-SAME: accumulator = !ondsp.acc<storage = i64, frac = 62, signed, update_overflow = saturate>
// Q31-SAME: dst = #ondsp.fixed<signed, storage = i32, frac = 31>
// Q31-SAME: numeric = #ondsp.fixed<signed, storage = i32, frac = 31>
// Q31-SAME: rounding = #ondsp.rounding<nearest_even>

// Q31-DEFAULT: invalid_q31_fir_stream_default.ox:6:10: error: a Q31 fir_stream requires an explicit accumulator, rounding, and overflow policy
